####################################################
# Combine all available data for seroprevalence from
# 2022-02-11 onwards
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

# Blood donor seroprevalence
donor_csv <- file.path(
  "data",
  "serological-data",
  "donor2022.csv"
)
donor_df <- data.table::fread(donor_csv) |>
  select(-quarter) |>
  mutate(date = as.Date(date))

# Commercial lab pediatric seroprevalence
lab_csv <- file.path(
  "data",
  "serological-data",
  "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
)
lab_df <- data.table::fread(lab_csv)

## Extract only pediatric data from Round 31 onwards
## Represent data using middle of date range
cols <- c(
  "n [Anti-N, 0-17 Years Prevalence]",
  "Rate (%) [Anti-N, 0-17 Years Prevalence]"
)

lab_pediatric <- lab_df |>
  filter(`Date Range of Specimen Collection` != "", Round > 30) |>
  select(Site, `Date Range of Specimen Collection`, all_of(cols)) |>
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

lab_pediatric <- lab_pediatric |>
  select(
    region = Site, date = mid_date,
    n = `n [Anti-N, 0-17 Years Prevalence]`,
    rate = `Rate (%) [Anti-N, 0-17 Years Prevalence]`
  ) |>
  filter(rate < 100) |>
  mutate(age = "0-17")

# Stack lab and donor
comb_df <- bind_rows(donor_df, lab_pediatric) |>
  filter(!region %in% c("DC", "PR"))
with(comb_df, table(region, age))

comb_df |>
  filter(region != "US") |>
  ggplot() +
  geom_point(aes(x = date, y = rate, colour = age, size = n)) +
  facet_wrap(~region)

# Output by state
states <- data.table::fread("./data/fips_to_name.csv")
for (st in unique(comb_df$region)) {
  dat_st <- comb_df |>
    filter(region == st) |>
    select(-region)
  # Add NAs where needed
  dat_st_fill <- dat_st |>
    tidyr::complete(
      date = unique(dat_st$date),
      age = unique(dat_st$age)
    ) |>
    arrange(date, age)
  stname <- states$stname[states$stusps == st]
  stname <- stringr::str_replace_all(stname, " ", "_")
  outfile <- file.path(
    "./data/serological-data/fitting-2022",
    glue::glue("{stname}_sero.csv")
  )
  dat_st_fill |>
    data.table::fwrite(outfile)
}
