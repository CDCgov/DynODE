#################################################
# Fit immunity waning model to a waning curve
# based on a logistic equation.
# The immunity waning model is based on ODE
# compartmental model, where you specify number
# of compartments, the flow rates among compartment
# (in a chain way) and the amount of immunity in
# each compartment.
# Optimal parameters are chosen based on genetic
# algorithm.
#################################################

source("data_manipulation_scripts/waning_helper.R")
pacman::p_load(GA)
pacman::p_load(doParallel, doRNG)
theme_set(theme_bw(base_size = 14))
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.grid.minor = element_blank(),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

# Waning equation is logistic curve based on Pfizer's 6 month data
# waning_equation <- function(x) 1 / (1 + exp(-(2.46 - 0.2 * x / 30)))
waning_equation <- function(x) 1 / (1 + exp(-(2.396 - 0.0143 * x)))
waning_df <- data.frame(time = 0:1000) |>
  mutate(eqn_immunity = ifelse(time <= 21,
    1.0, waning_equation(time - 21) / waning_equation(0)
  ))

# Some global variables
ncomp <- 5
init_protection <- 1.0

fit <- function(run_length, num_compartment) {
  # Preset run length and number of compartments
  function(waning_rates, protection, waning_df) {
    # Return the MSE of the fit wrt to waning_df for given
    # waning_rates and compartment protection values
    out <- make_waning_model(
      run_length = run_length,
      num_compartment = num_compartment,
      waning_rates = waning_rates,
      protection = protection,
      plot = FALSE
    )
    result <- out[[1]] |> left_join(waning_df, by = "time")
    result <- result |>
      mutate(
        se = (immunity - eqn_immunity)^2,
        ae = abs(immunity - eqn_immunity)
      ) |>
      summarise(
        mse = mean(se),
        mae = mean(ae)
      )

    return(result$mae)
  }
}

# Create objective function for GA
fit1 <- fit(1000, ncomp)
fit1_vector <- function(vec, init_protection, waning_df) {
  # Take vector of values, decompose to waning days and
  # protection multipliers, and then get MSE in return
  days <- vec[1:(ncomp - 1)]
  multipliers <- vec[ncomp:length(vec)]

  waning_rates <- c(1 / days, 0)
  multiplier_cumprod <- cumprod(multipliers)
  protection <- init_protection * c(1, multiplier_cumprod)

  return(-fit1(waning_rates, protection, waning_df))
}

# Some starting values to guide the genetic algorithm
v <- c(50, 50, 50, 50, 0.8, 0.8, 0.8, 0.8)

# Start GA
ga_object <- ga(
  type = "real-valued",
  fitness = fit1_vector,
  lower = rep(0, (ncomp - 1) * 2),
  upper = c(rep(300, ncomp - 1), rep(1, ncomp - 1)),
  parallel = TRUE,
  maxiter = 1000,
  run = 100,
  popSize = 100,
  pmutation = 0.2,
  suggestions = v,
  optim = TRUE, # incorporate BFGS optim in the ga
  seed = 4327,
  init_protection = init_protection,
  waning_df = waning_df
)

# Extract solutions
sol <- ga_object@solution
m <- matrix(sol, nrow = 1)

# Process results of the optimal parameters
waning_rates <- c(1 / sol[1:(ncomp - 1)], 0)
multipliers <- cumprod(sol[ncomp:length(sol)])
protection <- init_protection * c(1, multipliers)

result <- make_waning_model(
  run_length = 1000,
  num_compartment = ncomp,
  waning_rates = waning_rates,
  protection = protection
)

# Plot immunity profile based on optimal parameters vs equation
result[[1]]$immunity <- as.vector(result[[1]]$immunity)
res_time_immune <- result[[1]] |>
  select(time, immunity)
waning_df1 <- waning_df |>
  left_join(res_time_immune, by = "time") |>
  tidyr::pivot_longer(-time, names_to = "type", values_to = "immunity")

ggplot() +
  geom_line(aes(x = time, y = immunity, colour = type), data = waning_df1) +
  geom_point(aes(x = day, y = protection),
    data = result[[2]],
    colour = "grey20"
  ) +
  scale_colour_manual(
    name = "", values = c("black", "red"),
    labels = c("Conceptual", "Fitted")
  ) +
  labs(
    x = "Days since immunity acquisition",
    y = "Protection from immunity"
  ) +
  theme(
    legend.title = element_blank(),
    legend.position = c(0.99, 0.99),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.background = element_rect(fill = alpha("white", 0.4))
  )

ggsave("immunity_profile.png", width = 1500, height = 1200, units = "px")
