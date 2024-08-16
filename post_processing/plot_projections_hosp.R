pacman::p_load(dplyr, ggplot2, lubridate, tidyr, glue, stringr, grates)
pacman::p_load(cowplot)
pacman::p_load(extrafont)
pacman::p_load(geofacet)
theme_set(theme_bw(base_size = 12))
theme_update(
  strip.background = element_blank(),
  # strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.25), colour = "#D1D3D4"),
  panel.grid.minor = element_line(linewidth = rel(0.125)),
  panel.grid.minor.y = element_blank(),
  panel.border = element_rect(linewidth = rel(0.5)),
  axis.ticks = element_line(linewidth = rel(0.25))
)

retrieve_history_hosp <- function(state) {
  if (state == "US") {
    state_name <- "United States"
  } else {
    state_name <- state.name[state.abb == state]
  }
  state_name <- stringr::str_replace(state_name, " ", "_")
  f <- file.path(
    "/input/data", "hospitalization-data",
    glue("{state_name}_hospitalization.csv")
  )
  return(data.table::fread(f))
}

projection_raw <- data.table::fread("./output/projections_2407_2507_v0.csv")
total_projection <- projection_raw |>
  filter(!is.na(pred_hosp_0_17)) |>
  filter(date <= ymd("2025-06-29")) |> # last full epiweek end date + 1
  mutate(date = as.Date(date) - 1 + 7, # 7 days delay in hospitalization
         week_end_date = as_epiweek(date) |> date_end()) |>
  group_by(date = week_end_date, state, scenario, chain_particle) |>
  summarise(across(contains("hosp"), sum), .groups = "drop")
  # pivot_longer(contains("pred_hosp"), names_to = "age", names_prefix = "pred_hosp_", values_to = "hosp")

total_projection_all <- total_projection |>
  mutate(hosp_all = pred_hosp_0_17 + pred_hosp_18_49 + pred_hosp_50_64 + `pred_hosp_65+`) |>
  separate(scenario, into = c("vs", "it", "ie", "ve")) |>
  mutate(vs = str_remove(vs, "vs"),
         it = str_remove(it, "it"), 
         ve = str_remove(ve, "ve"), 
         ie = str_remove(ie, "ie")) |>
  mutate(it = factor(it, levels = c("aug", "sep", "oct", "nov", "dec", "non")))
  
# total_projection_mean <- total_projection |>
#   group_by(date, state, scenario, chain_particle) |>
#   mutate(hosp = sum(hosp)) |>
#   group_by(date, state, scenario) |>
#   summarise(hosp_mean = mean(hosp),
#             hosp_median = median(hosp),
#             hosp_lci = quantile(hosp, 0.025),
#             hosp_uci = quantile(hosp, 0.975)) |>
#   separate(scenario, into = c("vs", "it", "ie", "ve")) |>
#   mutate(vs = str_remove(vs, "vs"),
#          it = str_remove(it, "it"), 
#          ve = str_remove(ve, "ve"), 
#          ie = str_remove(ie, "ie"))

states <- projection_raw$state |> unique()
states_history <- purrr::map_dfr(states, retrieve_history_hosp, .id = "state")
states_history$state <- states[as.numeric(states_history$state)]
total_history <- states_history |>
  filter(date >= ymd("2023-07-01")) |> # limit historical length to display
  group_by(date, state) |>
  summarise(hosp = sum(hosp) * 7)

ITlab <- function(string) paste0("Intro time: ", string)
pdf("output/projections_2024_v0.pdf", width = 10, height = 10,
    onefile = TRUE)
for (st in state.abb) {
  p <- ggplot() +
    geom_line(aes(x = date, y = hosp), data = total_history |> filter(state == st)) +
    geom_line(aes(x = date, y = hosp_all, group = interaction(vs, chain_particle),
                  colour = vs),
              data = total_projection_all |> filter(state == st),
              alpha = 0.05) +
    facet_wrap(~ it, ncol = 2, labeller = as_labeller(ITlab)) +
    scale_x_date(breaks = ymd(c("2023-07-01", "2023-10-01", "2024-01-01", "2024-04-01", 
                                "2024-07-01", "2024-09-01", "2024-11-01", "2025-01-01", 
                                "2025-03-01", "2025-05-01", "2025-07-01")),
                 date_labels = "%b\n%y", expand = expansion(mult = 0.01)) +
    scale_colour_manual(values = c("deepskyblue", "blue3"),
                        labels = c("No booster", "Booster for all")) +
    guides(colour = guide_legend(override.aes = list(alpha = 1))) +
    labs(y = "Weekly hospitalizations",
         title = st) +
    theme(
      legend.position = "inside",
      legend.location = "plot",
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "#ffffff88"),
      legend.justification = c(1, 1.075),
      legend.title = element_blank(),
      panel.grid.minor.x = element_blank(),
      # legend.title = element_blank(),
      axis.title.x = element_blank()
    )
  print(p)
}
dev.off()
