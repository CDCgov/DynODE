pacman::p_load(dplyr, data.table, lubridate, tidyr, ggplot2)
theme_set(theme_bw(base_size = 13))
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

sero_flus <- fread("./data/abm-data/serology_flus.csv")
sero_abm <- fread("./data/abm-data/serology_abm.csv")

crude_auc <- function(days, values, rebase = FALSE) {
  if (rebase) values <- values - values[1]
  xout <- seq(min(days), max(days), by = 1)
  yout <- approx(days, values, xout)$y
  return(sum(yout))
}

sample_inf_hist <- function(df, probs) {
  retain <- rbinom(nrow(df), 1, probs)
  return(df[retain == 1, ])
}

calc_sero <- function(inf_df) {
  first_infections <- inf_df |>
    group_by(pid, age) |>
    summarise(infected_time = min(infected_time))

  serology <- first_infections |>
    group_by(age, time = infected_time) |>
    summarise(n = n()) |>
    complete(time = 0:874, fill = list(n = 0)) |>
    left_join(pop_age, by = "age") |>
    mutate(
      seroprevalence = cumsum(n) / pop * 100,
      Site = "ABM_scaled",
      date = ymd("2020-02-10") + time
    )

  return(serology)
}

plot_sero <- function(abm_df) {
  ggplot() +
    geom_line(aes(x = mid_date, y = value, colour = Site),
      data = sero_flus |> filter(Site == "US")
    ) +
    geom_line(aes(x = date, y = seroprevalence, colour = Site),
      data = abm_df
    ) +
    geom_vline(
      xintercept = ymd(c("2021-11-15", "2022-02-10")),
      lty = 2
    ) +
    facet_wrap(~age) +
    scale_x_date(date_breaks = "6 months", date_labels = "%m/%y") +
    theme(axis.title.x = element_blank()) +
    labs(y = "Seroprevalence %")
}

#### ABM infection history
age_df <- data.table::fread("./data/abm-data/sim_ages.txt")
db_path <- "./data/abm-data/sim_data_0.sqlite"
db_out_path <- "./data/abm-data/sim_data_scaled_us.sqlite"
file.copy(db_path, db_out_path, overwrite = TRUE)
con <- dbConnect(SQLite(), db_out_path)
dbListTables(con)
q <- dbSendQuery(
  con,
  "SELECT inf, inf_owner_id, infected_time, strain FROM infection_history"
)
query_inf_history <- dbFetch(q)
dbClearResult(q)
# dbDisconnect(con)

age_df <- age_df |>
  mutate(age = case_when(
    age <= 17 ~ "0-17",
    age <= 49 ~ "18-49",
    age <= 64 ~ "50-64",
    TRUE ~ "65+"
  ))
pop_age <- age_df |>
  group_by(age) |>
  summarise(pop = n())
infection_history <- query_inf_history |>
  left_join(age_df, by = join_by(inf_owner_id == pid)) |>
  rename(pid = inf_owner_id) |>
  mutate(
    date = ymd("2020-02-10") + infected_time,
    strain = ifelse(strain == "OMICRON", "OMICRON", "PREOM")
  )

#### Pre-Omicron scaling
preom_obs <- sero_flus |>
  filter(Site == "US") |>
  select(date = mid_date, age, value) |>
  filter(date <= ymd("2021-11-15"))
preom_start <- min(preom_obs$date)
preom_stop <- max(preom_obs$date)

preom_obs_auc <- preom_obs |>
  group_by(age) |>
  summarise(
    auc_obs = crude_auc(date, value, rebase = FALSE),
    last_obs = value[date == preom_stop]
  )

preom_abm_auc <- sero_abm |>
  filter(date >= preom_start, date <= preom_stop) |>
  group_by(age) |>
  summarise(
    auc_abm = crude_auc(date, seroprevalence, rebase = FALSE),
    last_abm = seroprevalence[date == preom_stop],
    first_abm = 0
  )

preom_scale <- preom_obs_auc |>
  left_join(preom_abm_auc, by = "age") |>
  mutate(
    scale_preom_auc = auc_obs / auc_abm,
    scale_om_lin = (last_obs - first_abm) / (last_abm - first_abm)
  ) |>
  select(-contains("obs"), -contains("abm"))

#### Sample preom
preom_samp <- infection_history |>
  left_join(preom_scale, by = "age")
probs <- ifelse(
  preom_samp$strain == "OMICRON",
  1,
  preom_samp$scale_preom_auc # auc scale is better fit
)
preom_samp <- sample_inf_hist(infection_history, probs)
preom_sero <- calc_sero(preom_samp)

plot_sero(preom_sero) # Matches well on first dash line

#### Omicron scaling
om_obs <- sero_flus |>
  filter(Site == "US") |>
  select(date = mid_date, age, value) |>
  filter(date >= ymd("2021-11-15"), date <= ymd("2022-02-11"))
om_start <- min(om_obs$date)
om_stop <- max(om_obs$date)

om_obs_auc <- om_obs |>
  group_by(age) |>
  summarise(
    auc_obs = crude_auc(date, value, rebase = TRUE),
    last_obs = value[date == om_stop]
  )

om_abm_auc <- preom_sero |>
  filter(date >= om_start, date <= om_stop) |>
  group_by(age) |>
  summarise(
    auc_abm = crude_auc(date, seroprevalence, rebase = TRUE),
    last_abm = seroprevalence[date == om_stop],
    first_abm = seroprevalence[date == om_start]
  )

om_scale <- om_obs_auc |>
  left_join(om_abm_auc, by = "age") |>
  left_join(pintersect, by = "age") |>
  mutate(
    scale_om_auc = auc_obs / auc_abm / (1 - p),
    scale_om_lin = (last_obs - first_abm) / (last_abm - first_abm) # / (1 - p)
  ) |>
  select(-contains("obs"), -contains("abm"))

#### Sample pre- and post-om
om_samp <- preom_samp |>
  left_join(om_scale, by = "age")
probs <- ifelse(
  om_samp$strain == "OMICRON",
  om_samp$scale_om_lin, # linear scale is better here
  1
)
om_samp <- sample_inf_hist(preom_samp, probs)
om_sero <- calc_sero(om_samp)

plot_sero(om_sero)

ggsave("./fig/scaled_seroprevalence.png",
  width = 2800, height = 1800,
  units = "px"
)

#### write to database
sql <- "CREATE TABLE retention(
          inf TEXT PRIMARY KEY,
          retain INTEGER
        )"
dbExecute(con, sql)
dbReadTable(con, "retention")

retention <- infection_history |> select(inf)
retention$retain <- as.integer(retention$inf %in% om_samp$inf)

dbAppendTable(con, "retention", retention)
dbGetQuery(con, "SELECT SUM(retain) AS sum FROM retention")
dbDisconnect(con)
