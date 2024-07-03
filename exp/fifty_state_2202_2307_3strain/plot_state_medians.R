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

medians_df <- data.table::fread("./output/medians_v15a.csv")
medians_long <- medians_df |>
  pivot_longer(-state, names_to = "parameter") |>
  mutate(parameter = case_when(
    parameter == "INTRODUCTION_PERCS_0" ~ "INTRO PCT (BA2/4/5)",
    parameter == "INTRODUCTION_PERCS_1" ~ "INTRO PCT (XBB1)",
    parameter == "INTRODUCTION_SCALES_0" ~ "INTRO SCALE (BA2/4/5)",
    parameter == "INTRODUCTION_SCALES_1" ~ "INTRO SCALE (XBB1)",
    parameter == "INTRODUCTION_TIMES_0" ~ "INTRO TIME (BA2/4/5)",
    parameter == "INTRODUCTION_TIMES_1" ~ "INTRO TIME (XBB1)",
    parameter == "STRAIN_R0s_0" ~ "R0 (BA1)",
    parameter == "STRAIN_R0s_1" ~ "R0 (BA2/4/5)",
    parameter == "STRAIN_R0s_2" ~ "R0 (XBB1)",
    # parameter == "STRAIN_R0s_0" ~ "R0 (XBB1)",
    # parameter == "STRAIN_R0s_1" ~ "R0 (XBB2)",
    # parameter == "STRAIN_R0s_2" ~ "R0 (JN1)",
    parameter == "ihr_0" ~ "IHR (0-17)",
    parameter == "ihr_1" ~ "IHR (18-49)",
    parameter == "ihr_2" ~ "IHR (50-64)",
    parameter == "ihr_3" ~ "IHR (65+)",
    parameter == "STRAIN_INTERACTIONS_0" ~ "INTERACTIONS BA1 <- BA1",
    parameter == "STRAIN_INTERACTIONS_3" ~ "INTERACTIONS BA1 <- BA2/4/5",
    parameter == "STRAIN_INTERACTIONS_6" ~ "INTERACTIONS BA1 <- XBB1",
    parameter == "STRAIN_INTERACTIONS_7" ~ "INTERACTIONS BA2/4/5 <- XBB1",
    # parameter == "STRAIN_INTERACTIONS_0" ~ "INTERACTIONS XBB1 <- XBB1",
    # parameter == "STRAIN_INTERACTIONS_3" ~ "INTERACTIONS XBB1 <- XBB2",
    # parameter == "STRAIN_INTERACTIONS_6" ~ "INTERACTIONS XBB1 <- JN1",
    # parameter == "STRAIN_INTERACTIONS_7" ~ "INTERACTIONS XBB2 <- JN1",
    TRUE ~ parameter
  ))

# View(medians_df |> select(state, SEASONALITY_AMPLITUDE, SEASONALITY_SECOND_WAVE, SEASONALITY_SHIFT))
# par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
# plot(medians_df[, 1:26], pch = 20)
# plot(medians_df[, c("SEASONALITY_AMPLITUDE", "SEASONALITY_SECOND_WAVE", "SEASONALITY_SHIFT")], pch = 20)

ggplot() +
  geom_violin(aes("", value),
    medians_long,
    draw_quantiles = 0.5,
  ) +
  geom_point(aes("", value), medians_long |> filter(state == "US"),
    colour = "red", alpha = 1.0
  ) +
  scale_colour_manual(
    name = "", values = c("gray50", "red"),
    label = c("States", "US")
  ) +
  facet_wrap(~parameter, scales = "free") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )
