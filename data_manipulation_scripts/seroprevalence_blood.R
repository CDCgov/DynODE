####################################################
# Process the blood donor seroprevalence data from CDC
# Because blood donor is less representative of actual
# population, we adjust the blood donor seroprevalence.
# Compare the ratio between commercial lab seroprevalence
# with donor seroprevalence in 2020-2021, then apply the
# same ratio to donor 2022 data to obtain commercial lab
# equivalent of 2022 (no commercial lab data in 2022).
# Plot the output and write out to CSV
####################################################

pacman::p_load(dplyr, ggplot2, lubridate, tidyr, glue)
theme_set(theme_bw(base_size = 13))
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

# Input
## Nationwide commercial lab seroprevalence
lab_csv <- file.path(
  "data",
  "serological-data",
  "serology_flus.csv"
)
lab_df <- data.table::fread(lab_csv) |>
  filter(Site == "US") |>
  filter(year(mid_date) <= 2021)

## 2020-2021 Blood donor seroprevalence
donor1_csv <- file.path(
  "data",
  "serological-data",
  "2020-2021_Nationwide_Blood_Donor_Seroprevalence_Survey_Infection-Induced_Seroprevalence_Estimates_20231018.csv" # nolint: line_length_linter.
)
donor1_df <- data.table::fread(donor1_csv) |>
  filter(`Region Abbreviation` == "All")

### Manipulate into long form
donor1_long <- donor1_df |>
  select(
    date = `Median\nDonation Date`,
    contains("Years Prevalence") & contains("Rate")
  ) |>
  mutate(date = mdy(date)) |>
  pivot_longer(-date,
    names_to = "age",
    names_pattern = "Rate \\(%\\) \\[(.*) Years Prevalence\\]"
  )

donor1_long <- donor1_long |>
  mutate(age = case_when(
    age == "16-29" ~ "18-49",
    age == "30-49" ~ "18-49",
    TRUE ~ age
  )) |>
  group_by(date, age) |>
  summarise(value = mean(value))

## 2022 Blood donor seroprevalence
donor2_csv <- file.path(
  "data",
  "serological-data",
  "2022_Nationwide_Blood_Donor_Seroprevalence_Survey_Combined_Infection-_and_Vaccination-Induced_Seroprevalence_Estimates_20240307.csv" # nolint: line_length_linter.
)
donor2_df <- data.table::fread(donor2_csv) |>
  filter(
    `Geographic Identifier` == "USA",
    Race == "Overall",
    Sex == "Overall",
    Indicator == "Past infection with or without vaccination"
  ) |>
  filter(Age != "Overall")

### Manipulate age and mean+-CI
donor2_df <- donor2_df |>
  mutate(age = case_when(
    Age == "16 to 29" ~ "18-49",
    Age == "30 to 49" ~ "18-49",
    Age == "50 to 64" ~ "50-64",
    TRUE ~ "65+"
  )) |>
  group_by(date = `Time Period`, age) |>
  summarise(
    value = mean(`Estimate % (weighted)`),
    lci = mean(`2.5%`),
    uci = mean(`97.5%`)
  )

donor2_df <- donor2_df |>
  mutate(
    quarter = stringr::str_remove(date, "2022 Quarter ") |> as.numeric(),
    date = ymd("2022-02-15")
  )
month(donor2_df$date) <- month(donor2_df$date) + (donor2_df$quarter - 1) * 3

# Adjust blood donor 2022 seroprevalence
## Take relationship between commercial lab and blood donor
## in pre-2022, then apply to blood donor 2022 to obtain
## commercial lab equivalent of 2022 (deeming commercial lab)
## as more representative of actual seroprevalence
donor2_adj_df <- data.frame()
for (ag in c("18-49", "50-64", "65+")) {
  donor1_age <- donor1_long |> filter(age == ag)
  donor2_age <- donor2_df |> filter(age == ag)
  lab_age <- lab_df |> filter(age == ag)

  # match lab dates with donor dates
  lab_prev <- approx(lab_age$mid_date,
    lab_age$value,
    xout = donor1_age$date
  )$y

  # calculate ratio in between lab and donor in logit space
  donor1_lprev <- arm::logit(donor1_age$value / 100)
  lab_lprev <- arm::logit(lab_prev / 100)
  m <- mean(lab_lprev - donor1_lprev, na.rm = TRUE)

  # apply the ratio to donor 2022 data
  donor2_mat <- as.matrix(
    donor2_age[, c("value", "lci", "uci")] / 100
  )
  adj_donor2_mat <- (arm::logit(donor2_mat) + m) |>
    arm::invlogit()

  # collect
  donor2_age[, c("value", "lci", "uci")] <- adj_donor2_mat * 100
  donor2_adj_df <- bind_rows(donor2_adj_df, donor2_age)
}

# simple visualization
donor2_adj_df |>
  ggplot() +
  geom_line(aes(x = date, y = value, colour = age))

# calculate logit scale mean and sd
donor2_adj_fin <- donor2_adj_df |>
  mutate(across(value:uci, ~ arm::logit(.x / 100), .names = "logit_{.col}")) |>
  mutate(logit_sd = ((logit_uci - logit_value) + (logit_value - logit_lci)) /
    (2 * 1.96)) |>
  select(-logit_lci, -logit_uci) |>
  rename(logit_mean = logit_value)

data.table::fwrite(donor2_adj_fin, "data/serological-data/donor2022.csv")
