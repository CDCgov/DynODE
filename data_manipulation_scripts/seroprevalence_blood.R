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
## Using the derived/adjusted version out of seroprevalence_rnd1to30_adj.R
lab_csv <- file.path(
  "data",
  "serological-data",
  "seroprevalence_50states.csv"
)
lab_df <- data.table::fread(lab_csv)
lab_df2021 <- data.table::fread(lab_csv) |>
  filter(year(date) < 2022)

## 2020-2021 Blood donor seroprevalence
donor1_csv <- file.path(
  "data",
  "serological-data",
  "2020-2021_Nationwide_Blood_Donor_Seroprevalence_Survey_Infection-Induced_Seroprevalence_Estimates_20231018.csv" # nolint: line_length_linter.
)
donor1_df <- data.table::fread(donor1_csv)

### Manipulate into long form and extract n and rates
### Then pivot back to wide form with n and rates as column
donor1_long <- donor1_df |>
  rename(region = `Region Abbreviation`) |>
  select(
    date = `Median\nDonation Date`,
    region,
    contains("Years Prevalence") & (starts_with("Rate") | starts_with("n"))
  ) |>
  mutate(date = mdy(date)) |>
  pivot_longer(-c(date, region),
    names_to = c("metric", "age"),
    names_pattern = "^(.*) \\[(.*) Years Prevalence\\]"
  ) |>
  mutate(age = trimws(age, which = "both"))

donor1_wide <- donor1_long |>
  mutate(metric = ifelse(metric == "n", "n", "rate")) |>
  pivot_wider(
    id_cols = c("date", "region", "age"),
    names_from = "metric", values_from = "value"
  )

### Consolidate 18-49 age group (assume 16-29 == 18-29)
donor1_wide <- donor1_wide |>
  mutate(age = case_when(
    age == "16-29" ~ "18-49",
    age == "30-49" ~ "18-49",
    TRUE ~ age
  )) |>
  group_by(region, date, age) |>
  summarise(
    rate = sum(rate * n) / sum(n),
    n = sum(n)
  )

### Consolidate by states (Because 2022 only has states)
### Some states have multiple sites, each site has different dates,
### standardize them by aligning year-month
donor1_region <- donor1_wide |>
  mutate(
    region = stringr::str_extract(region, "^([A-Z]+)"),
    region = ifelse(region == "A", "US", region),
    month = month(date),
    year = year(date)
  ) |>
  group_by(region, year, month, age) |>
  summarise(
    rate = sum(rate * n) / sum(n),
    n = sum(n),
    date = mean(date)
  )

ggplot(donor1_region) +
  geom_line(aes(x = date, y = rate, colour = age)) +
  facet_wrap(~region)

## 2022 Blood donor seroprevalence
donor2_csv <- file.path(
  "data",
  "serological-data",
  "2022_Nationwide_Blood_Donor_Seroprevalence_Survey_Combined_Infection-_and_Vaccination-Induced_Seroprevalence_Estimates_20240307.csv" # nolint: line_length_linter.
)
donor2_df <- data.table::fread(donor2_csv) |>
  filter(
    Race == "Overall",
    Sex == "Overall",
    Indicator == "Past infection with or without vaccination"
  ) |>
  filter(Age != "Overall")

### Rate and n by age, assuming 16-29 = 18-29
donor2_region <- donor2_df |>
  mutate(
    age = case_when(
      Age == "16 to 29" ~ "18-49",
      Age == "30 to 49" ~ "18-49",
      Age == "50 to 64" ~ "50-64",
      TRUE ~ "65+"
    ),
    rate = `Estimate % (weighted)`,
    n = `n (Unweighted)`
  ) |>
  group_by(region = `Geographic Area`, date = `Time Period`, age) |>
  summarise(
    rate = sum(rate * n) / sum(n),
    n = sum(n)
  )

### Apply state abbreviation, and fix date for donor2
donor2_region <- donor2_region |>
  ungroup() |>
  mutate(
    region2 = state.abb[match(region, state.name)],
    region2 = ifelse(region == "Overall", "US", region2)
  ) |>
  filter(!is.na(region2)) |>
  mutate(
    quarter = stringr::str_remove(date, "2022 Quarter ") |> as.numeric(),
    date = ymd("2022-02-15")
  ) |>
  select(-region) |>
  rename(region = region2)
month(donor2_region$date) <- month(donor2_region$date) + (donor2_region$quarter - 1) * 3 # nolint: line_length_linter.

# Adjust blood donor 2022 seroprevalence
## Take relationship between commercial lab and blood donor
## in pre-2022, then apply to blood donor 2022 to obtain
## commercial lab equivalent of 2022 (deeming commercial lab)
## as more representative of actual seroprevalence
states <- donor2_region$region |> unique()
states <- setdiff(states, "ND") # excluding ND here because no donor1 data
donor2_adj_df <- data.frame()
for (st in states) {
  for (ag in c("18-49", "50-64", "65+")) {
    donor1_age <- donor1_region |>
      filter(region == st, age == ag)
    donor2_age <- donor2_region |>
      filter(region == st, age == ag)
    lab_age <- lab_df2021 |>
      filter(Site == st, age == ag)

    # match lab dates with donor dates
    lab_prev <- approx(lab_age$date,
      lab_age$infected,
      xout = donor1_age$date
    )$y

    # calculate ratio in between lab and donor in logit space
    donor1_lprev <- arm::logit(donor1_age$rate / 100)
    lab_lprev <- arm::logit(lab_prev)
    diff <- lab_lprev - donor1_lprev
    cond <- !is.na(diff) & is.finite(diff)
    diff <- diff[cond]
    n <- donor1_age$n[cond]
    m <- sum(diff * n) / sum(n)

    # apply the ratio to donor 2022 data
    adj_donor2_prev <- (arm::logit(donor2_age$rate / 100) + m) |>
      arm::invlogit()

    # collect
    donor2_age$rate <- adj_donor2_prev * 100
    donor2_age$m <- m
    donor2_adj_df <- bind_rows(donor2_adj_df, donor2_age)
  }
}

# Add ND without adjustment here (only data we have about ND)
donor2_nd <- donor2_region |>
  filter(region == "ND")
donor2_adj_stack <- bind_rows(donor2_adj_df, donor2_nd)

# "Lifting" the seroprevalence up so that feb donor matches with
# feb lab
lab_feb <- lab_df |>
  filter(date > ymd("2022-02-01")) |>
  select(-date)
donor2_adj_fin <- donor2_adj_stack |>
  left_join(lab_feb, by = c("region" = "Site", "age")) |>
  group_by(region, age) |>
  mutate(lift = arm::logit(infected) - arm::logit(rate[quarter == 1] / 100)) |>
  mutate(
    lifted_rate = arm::logit(rate / 100) + lift,
    lifted_rate = arm::invlogit(lifted_rate)
  ) |>
  mutate(rate = coalesce(lifted_rate * 100, rate)) |>
  select(-c(lift, m, lifted_rate))

# simple visualization
donor2_adj_fin |>
  filter(region != "US") |>
  ggplot() +
  geom_point(aes(x = date, y = rate, colour = age, size = n)) +
  facet_wrap(~region)

data.table::fwrite(donor2_adj_fin, "data/serological-data/donor2022.csv")
