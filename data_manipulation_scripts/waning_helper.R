pacman::p_load(dplyr, deSolve, ggplot2, tidyr)

logit <- function(x) log(x / (1 - x))
invlogit <- function(x) exp(x) / (1 + exp(x))

waning_ode <- function(time, states, parameters) {
  # A simple waning model with chain of W compartments
  with(parameters, {
    l <- length(states)
    dw <- -waning_rates * states
    dw[2:l] <- dw[2:l] - dw[1:(l - 1)]

    return(list(dw))
  })
}

make_waning_model <- function(run_length, num_compartment, waning_rates,
                              protection, plot = TRUE) {
  # Based on input parameters to the compartmental waning model,
  # calculate solution (daily) and optionally plot the solutions
  parameters <- list(
    waning_rates = waning_rates,
    protection = protection
  )
  initial_states <- c(
    W = c(1, rep(0, num_compartment - 1))
  )
  times <- seq(0, run_length)

  solution <- ode(
    y = initial_states,
    times = times,
    func = waning_ode,
    parms = parameters
  ) |> as.data.frame()
  solution$immunity <- as.matrix(solution[, -1]) %*% parameters$protection

  parameter_df <- as.data.frame(parameters) |>
    mutate(day = c(0, cumsum(1 / waning_rates[1:(n() - 1)])))

  if (plot) {
    gg <- ggplot() +
      geom_line(aes(x = time, y = immunity), data = solution, colour = "blue") +
      geom_line(aes(x = time, y = eqn_immunity),
        data = waning_df,
        colour = "red"
      ) +
      geom_point(aes(x = day, y = protection),
        data = parameter_df,
        colour = "grey20"
      ) +
      theme_bw()
    print(gg)
  }

  return(list(solution, parameter_df))
}
