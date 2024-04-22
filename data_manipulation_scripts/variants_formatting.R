####################################################
# Take CDC's variant proportion data, and reclassify
# the strains to the "major epoch" strains, and then
# recalculate the variant proportion by removing all
# the "others".
# Write out to 50-state csv based on HHS region
####################################################

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

# start and end date
date_start <- ymd("2022-02-13")
date_end <- ymd("2024-05-01")

# regions lookup
hhs <- c(
  4, 10, 9, 6, 9, 8, 1, 3, 4, 4, 9, 10, 5, 5, 7,
  7, 4, 6, 1, 3, 1, 5, 5, 4, 7, 8, 7, 9, 1, 2,
  6, 2, 4, 8, 5, 6, 10, 3, 1, 4, 8, 4, 6, 8, 1,
  4, 10, 3, 5, 8
)
hhs_lookup <- data.frame(state = state.abb, hhs = hhs)

# process raw data taking only biweekly "weighted" data which
# is more systematic
dat_csv <- file.path(
  "data",
  "variant-data",
  "SARS-CoV-2_Variant_Proportions_20240421.csv"
)
dat <- data.table::fread(dat_csv)

dat_processed <- dat |>
  filter(
    modeltype == "weighted",
    share != 0
  ) |>
  mutate(week_ending = as.Date(mdy_hms(week_ending))) |>
  select(
    region = usa_or_hhsregion, date = week_ending, variant,
    share
  )

# subset to date of interests and only selecting variant with
# at least 1% appearance in its history
dates <- unique(dat_processed$date)
dat_subset <- dat_processed |>
  filter(date >= date_start, date <= date_end) |>
  group_by(variant) |>
  mutate(max_share = max(share)) |>
  filter(max_share > 0.01)

# reclass lookup:
# 0: omicron and pre-omicron
# 1: BA2/4/5 #nolint
# 2: XBB1
# 3: XBB2
# 4: JN1
variants <- unique(dat_subset$variant) |> sort()
strain_num <- c(
  0, 0, 1, 1, 2, 2, 4, 1, 1, 1,
  1, 1, 1, 9, 2, 2, 2, 3, 3, 3,
  9, 9, 9, 9, 3, 9, 9, 9, 9, 3,
  3, 3, 9, 9, 4, 4, 4, 4, 9, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 3, 3, 3, 3
)
var_lookup <- data.frame(variant = variants, strain = strain_num)

# reclassify to match current inference model
# removing others (9) from this
dat_reclass <- dat_subset |>
  left_join(var_lookup, by = "variant") |>
  filter(strain != 9) |>
  group_by(region, date, strain) |>
  summarize(share = sum(share), .groups = "drop_last") |>
  mutate(share = share / sum(share)) |>
  ungroup()
## fill in the zeros
dat_reclass_fill <- dat_reclass |>
  tidyr::complete(
    region = unique(dat_reclass$region),
    date = unique(dat_reclass$date),
    strain = 0:4,
    fill = list(share = 0)
  )

# visualize
ggplot() +
  geom_col(aes(x = date, y = share, fill = as.factor(strain)),
    data = dat_reclass_fill
  ) +
  facet_wrap(~region)

# output by state
states <- data.table::fread("./data/fips_to_name.csv") |>
  filter(stusps != "DC")
for (i in seq_len(nrow(states))) {
  stusps <- states$stusps[i]
  stname <- states$stname[i] |>
    stringr::str_replace_all(" ", "_")
  reg <- ifelse(
    st == "US",
    "USA",
    hhs_lookup$hhs[hhs_lookup$state == stusps] |> as.character()
  )

  outfile <- file.path(
    "./data/variant-data/fitting-2022",
    glue::glue("{stname}_strain_prop.csv")
  )

  df_out <- dat_reclass_fill |>
    filter(region == reg) |>
    arrange(date, strain)
  df_out |> data.table::fwrite(outfile)
}

# output the whole thing
data.table::fwrite(
  dat_reclass_fill,
  "./data/variant-data/variant_proportions_20220219_20240316.csv"
)
