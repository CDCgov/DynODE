####################################################
# Process the serology data downloaded from CDC
# Process the FL ABM output and extract serology information
# Plot the seroprevalence from actual observation vs ABM
# Write out processed serology data as CSVs
####################################################

pacman::p_load(dplyr, ggplot2, lubridate, tidyr, glue)
pacman::p_load(RSQLite)
theme_set(theme_bw(base_size = 13))
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

# Observed Serology
csv <- file.path(
  "data",
  "serological-data",
  "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
)
df <- data.table::fread(csv)
df

cols <- c(
  "Rate (%) [Anti-N, 0-17 Years Prevalence]",
  "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
  "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
  "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
)

df_flus <- df |>
  filter(Site %in% c("FL", "US")) |>
  select(
    Site, `Date Range of Specimen Collection`,
    all_of(cols)
  )

df_flus <- df_flus |>
  separate_wider_delim(
    `Date Range of Specimen Collection`,
    delim = "-", names = c("start_date", "end_date")
  ) |>
  mutate(
    end_date = mdy(end_date),
    start_date = ifelse(stringr::str_detect(start_date, ","),
      start_date, glue("{start_date}{year(end_date)}")
    ),
    start_date = mdy(start_date),
    mid_date = start_date + (end_date - start_date) / 2
  )


df_flus_long <- df_flus |>
  tidyr::pivot_longer(
    contains("Rate"),
    names_to = c("type", "age"),
    names_pattern = "Rate \\(%\\) \\[(.*), (.*) Years Prevalence.*\\]"
  )

ggplot(df_flus_long) +
  geom_smooth(aes(x = mid_date, y = value, colour = Site),
    span = 0.3
  ) +
  geom_vline(
    xintercept = ymd(c("2021-11-15", "2022-02-10")),
    lty = 2
  ) +
  facet_wrap(~age)

#### Florida ABM

age_df <- data.table::fread("./data/abm-data/sim_ages.txt")
con <- dbConnect(SQLite(), "./data/abm-data/sim_data_0.sqlite")
dbListTables(con)
q <- dbSendQuery(
  con,
  "SELECT inf_owner_id, infected_time FROM infection_history"
)
infection_history <- dbFetch(q)
dbClearResult(q)

colnames(infection_history)
infection_history <- age_df |>
  left_join(infection_history, by = join_by(pid == inf_owner_id))
head(infection_history)
first_inf <- infection_history |>
  group_by(pid, age) |>
  summarise(infected_time = min(infected_time), .groups = "drop") |>
  mutate(age = case_when(
    age <= 17 ~ "0-17",
    age <= 49 ~ "18-49",
    age <= 64 ~ "50-64",
    TRUE ~ "65+"
  ))
pop_age <- first_inf |>
  group_by(age) |>
  summarise(total = n())
serology <- first_inf |>
  group_by(age, time = infected_time) |>
  summarise(n = n()) |>
  filter(!is.na(time)) |>
  complete(time = 0:874, fill = list(n = 0)) |>
  left_join(pop_age, by = "age") |>
  mutate(seroprevalence = cumsum(n) / total)

# Combine
serology_abm <- serology |>
  mutate(
    Site = "FL_ABM",
    date = ymd("2020-02-10") + ddays(time),
    seroprevalence = seroprevalence * 100
  ) |>
  select(Site, age, date, seroprevalence)

ggplot() +
  geom_line(aes(x = mid_date, y = value, colour = Site),
    data = df_flus_long # , alpha = 0.3, span = 0.4
  ) +
  geom_line(aes(x = date, y = seroprevalence, colour = Site),
    data = serology_abm
  ) +
  geom_vline(
    xintercept = ymd(c("2021-11-15", "2022-02-10")),
    lty = 2
  ) +
  facet_wrap(~age) +
  scale_x_date(date_breaks = "6 months", date_labels = "%m/%y") +
  theme(axis.title.x = element_blank()) +
  labs(y = "Seroprevalence %")

ggsave("./fig/seroprevalence.png", width = 2800, height = 1800, units = "px")

####
data.table::fwrite(serology_abm, "serology_abm.csv")
data.table::fwrite(df_flus_long, "serology_flus.csv")
