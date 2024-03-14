####################################################
# Process the serology data downloaded from CDC
# Estimate seroprevalence at logit scale
# Estimate uncertainty at logit scale
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
  "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]",
  "Lower CI [Anti-N, 0-17 Years Prevalence]",
  "Lower CI [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
  "Lower CI [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
  "Lower CI [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]",
  "Upper CI [Anti-N, 0-17 Years Prevalence]",
  "Upper CI [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
  "Upper CI [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
  "Upper CI [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
)

df_all <- df |>
  select(
    Site, `Date Range of Specimen Collection`,
    all_of(cols)
  )

df_all <- df_all |>
  filter(`Date Range of Specimen Collection` != "") |>
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

df_all_long <- df_all |>
  tidyr::pivot_longer(
    -c(Site:end_date, mid_date),
    names_to = c("metric", "age"),
    names_pattern = "(.*) \\[Anti-N, (.*) Years Prevalence.*\\]"
  )

df_all_long <- df_all_long |>
  mutate(metric = case_when(
    metric == "Rate (%)" ~ "mean",
    metric == "Lower CI" ~ "lci",
    metric == "Upper CI" ~ "uci"
  )) |>
  tidyr::pivot_wider(
    names_from = "metric",
    values_from = "value"
  ) |>
  filter(mean <= 100)

adj_0_100 <- function(x) {
  x[x >= 100] <- 99.9
  x[x <= 0] <- 0.1
  return(x)
}

df_all_long <- df_all_long |>
  mutate(across(mean:uci, ~ arm::logit(adj_0_100(.x) / 100),
    .names = "logit_{.col}"
  )) |>
  mutate(
    logit_sd = ((logit_uci - logit_mean) + (logit_mean - logit_lci)) /
      (2 * 1.96)
  ) |>
  select(-logit_lci, -logit_uci)

ggplot(df_all_long |> filter(Site == "FL")) +
  geom_ribbon(aes(x = mid_date, ymin = lci, ymax = uci), alpha = 0.2) +
  geom_line(aes(x = mid_date, y = mean)) +
  facet_wrap(~age)

data.table::fwrite(df_all_long, "data/serological-data/serology_all.csv")
