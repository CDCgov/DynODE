library(pacman)
pacman::p_load(dplyr, ggplot2, lubridate)
pacman::p_get(data.table, force = FALSE)
pacman::p_get(tidyr, force = FALSE)
pacman::p_get(janitor, force = FALSE)
theme_set(theme_bw())
theme_update(text = element_text(family = "Open Sans"))
# Raw data found here:
# https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/sgxm-t72h/about_data # nolint: line_length_linter.
# if this line fails download the data and place it in the data folder.
dat_csv <- file.path(
  "data",
  "hospitalization-data",
  "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries_20240304.csv" # nolint: line_length_linter.
)
dat <- data.table::fread(dat_csv)

dat <- dat |>
  mutate(date = mdy_hms(date) |> as.Date())


# Pediatric
dat_us_pedia <- dat |>
  select(
    state, date,
    previous_day_admission_pediatric_covid_confirmed
  ) |>
  group_by(date) |>
  summarise(
    new_admission = sum(previous_day_admission_pediatric_covid_confirmed,
      na.rm = TRUE
    )
  )
dat_pedia <- dat_us_pedia |>
  mutate(agegroup = "0-17")


# Adult
dat_us_adult <- dat |>
  select(
    state, date,
    contains("previous_day_admission_adult_covid_confirmed")
  ) |>
  select(!contains("coverage")) |>
  group_by(date) |>
  summarise(across(-c(state), \(x) sum(x, na.rm = TRUE)))

dat_adult_long <- dat_us_adult |>
  tidyr::pivot_longer(contains("previous_day_admission_adult_covid_confirmed"),
    names_to = "age",
    names_pattern = "previous_day_admission_adult_covid_confirmed_(.*)",
    values_to = "new_admission"
  ) |>
  filter(!is.na(age), age != "unknown") |>
  tidyr::separate(age, into = c("agelo", "agehi"), convert = TRUE) |>
  mutate(agehi = coalesce(agehi, 100))

dat_60_69 <- dat_adult_long |>
  filter(agelo == 60)
dat_60_64 <- dat_60_69 |>
  mutate(
    agehi = 64,
    new_admission = new_admission / 2
  )
dat_65_69 <- dat_60_64 |>
  mutate(agelo = 65, agehi = 69)

dat_adult_long <- dat_adult_long |>
  filter(agelo != 60) |>
  bind_rows(dat_60_64, dat_65_69) |>
  arrange(date, agelo)

dat_adult_grouped <- dat_adult_long |>
  mutate(agegroup = case_when(
    agehi <= 49 ~ "18-49",
    agehi <= 64 ~ "50-64",
    TRUE ~ "65+"
  )) |>
  group_by(date, agegroup) |>
  summarise(new_admission = sum(new_admission))


# Populations
csv <- file.path(
  "data",
  "demographic-data",
  "population_rescaled_age_distributions",
  "United_States_country_level_age_distribution_85.csv"
)
pop_df <- data.table::fread(csv)
colnames(pop_df) <- c("age", "pop")
pop_agegroup <- pop_df |>
  mutate(agegroup = case_when(
    age <= 17 ~ "0-17",
    age <= 49 ~ "18-49",
    age <= 64 ~ "50-64",
    TRUE ~ "65+"
  )) |>
  group_by(agegroup) |>
  summarise(population = sum(pop))

# Combined
dat_agegroup <- bind_rows(dat_pedia, dat_adult_grouped) |>
  arrange(date, agegroup) |>
  left_join(pop_agegroup, by = "agegroup") |>
  mutate(incidence = new_admission / population * 1e5) # per 100000

dat_agegroup_subset <- dat_agegroup |>
  filter(date >= ymd("2020-08-02"), date <= ymd("2023-12-31")) |>
  mutate(
    week = epiweek(date),
    year = ifelse(week %in% 52:53 & month(date) == 1,
      year(date) - 1, year(date)
    )
  ) |>
  group_by(year, week, agegroup) |>
  mutate(
    new_admission_7 = mean(new_admission),
    incidence_7 = mean(incidence),
    week_end_date = max(date)
  )

dat_agegroup_weekly <- dat_agegroup_subset |>
  group_by(year, week, agegroup) |>
  summarise(
    new_admission = sum(new_admission),
    incidence = sum(incidence),
    week_end_date = max(date)
  )


# Plot
dat_agegroup |>
  filter(date >= ymd("2020-08-02"), date <= ymd("2023-12-31")) |>
  ggplot() +
  geom_line(aes(x = date, y = incidence, colour = agegroup)) +
  theme_bw()

dat_agegroup_subset |>
  ggplot() +
  geom_line(aes(x = date, y = incidence_7, colour = agegroup))

dat_agegroup_weekly |>
  ggplot() +
  geom_line(aes(x = week_end_date, y = incidence, colour = agegroup))


# Output
dat_agegroup_subset |>
  select(-population) |>
  data.table::fwrite("./data/hospitalization-data/hospital_200802_231231.csv")
