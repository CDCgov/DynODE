library(pacman)
pacman::p_load(dplyr, ggplot2, lubridate)
pacman::p_get(data.table, force = FALSE)
pacman::p_get(tidyr, force = FALSE)
pacman::p_get(janitor, force = FALSE)
theme_set(theme_bw())
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)
# Raw data found here:
# https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/sgxm-t72h/about_data # nolint: line_length_linter.
# if this line fails download the data and place it in the data folder.
dat_csv <- file.path(
  "data",
  "hospitalization-data",
  "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries_20240424.csv" # nolint: line_length_linter.
)
dat <- data.table::fread(dat_csv)

dat <- dat |>
  mutate(date = mdy_hms(date) |> as.Date())


# Pediatric
dat_pedia <- dat |>
  select(
    state, date,
    previous_day_admission_pediatric_covid_confirmed
  ) |>
  group_by(state, date) |>
  summarise(
    new_admission = sum(previous_day_admission_pediatric_covid_confirmed,
      na.rm = TRUE
    )
  )
dat_pedia <- dat_pedia |>
  mutate(agegroup = "0-17")


# Adult
dat_adult <- dat |>
  select(
    state, date,
    contains("previous_day_admission_adult_covid_confirmed")
  ) |>
  select(!contains("coverage")) |>
  group_by(state, date) |>
  summarise(across(everything(), \(x) sum(x, na.rm = TRUE)))

dat_adult_long <- dat_adult |>
  tidyr::pivot_longer(contains("previous_day_admission_adult_covid_confirmed"),
    names_to = "age",
    names_pattern = "previous_day_admission_adult_covid_confirmed_(.*)",
    values_to = "new_admission"
  ) |>
  filter(!is.na(age), age != "unknown") |>
  tidyr::separate(age, into = c("agelo", "agehi"), convert = TRUE) |>
  mutate(agehi = coalesce(agehi, 100))

## Because we want 50-64 age bin which cut through 60-69 in the hospitalization
## data, we assume that 60-69 is evenly distribute and we cut it in the middle
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

## Consolidating all the different age group into three
dat_adult_grouped <- dat_adult_long |>
  mutate(agegroup = case_when(
    agehi <= 49 ~ "18-49",
    agehi <= 64 ~ "50-64",
    TRUE ~ "65+"
  )) |>
  group_by(state, date, agegroup) |>
  summarise(new_admission = sum(new_admission))


# Combined (pediatric plus adult)
dat_agegroup <- bind_rows(dat_pedia, dat_adult_grouped) |>
  arrange(date, agegroup)

## Subsetting to Feb 2022 to end of 2023
dat_agegroup_subset <- dat_agegroup |>
  filter(state %in% state.abb) |>
  filter(date >= ymd("2022-02-13")) |>
  mutate(
    week = epiweek(date),
    year = ifelse(week %in% 52:53 & month(date) == 1,
      year(date) - 1, year(date)
    )
  ) |>
  group_by(year, week, state, agegroup) |>
  mutate(
    new_admission_7 = mean(new_admission),
    week_end_date = max(date)
  )

dat_agegroup_weekly <- dat_agegroup_subset |>
  filter(date == week_end_date)

## Summing up all states to form US-wide data
dat_us <- dat_agegroup_weekly |>
  group_by(date, agegroup) |>
  summarise(
    new_admission = sum(new_admission),
    new_admission_7 = sum(new_admission_7)
  ) |>
  mutate(state = "US")

## 50 states + US
dat_all <- bind_rows(dat_agegroup_weekly, dat_us) |>
  ungroup() |>
  select(state, date, agegroup, hosp = new_admission_7)

# Visualization
dat_all |>
  ggplot() +
  geom_line(aes(x = date, y = hosp, colour = agegroup)) +
  theme_bw() +
  scale_y_log10() +
  facet_wrap(~state)

# Output
states <- data.table::fread("./data/fips_to_name.csv")
for (st in unique(dat_all$state)) {
  dat_st <- dat_all |>
    filter(state == st) |>
    select(-state)
  stname <- states$stname[states$stusps == st]
  stname <- stringr::str_replace_all(stname, " ", "_")
  outfile <- file.path(
    "./data/hospitalization-data/",
    glue::glue("{stname}_hospitalization.csv")
  )
  dat_st |>
    data.table::fwrite(outfile)
}
