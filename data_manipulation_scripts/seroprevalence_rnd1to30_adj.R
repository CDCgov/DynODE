####################################################
# Script to process CDC serology data and produce
# seroprevalence estimates up to Round 30 (Feb 2022)
# Seroprevalence is adjusted according to Garcia-Carreras
# 2023 paper output. Because the output was only for
# all age, main bulk of this script is to find out how
# to estimate age-specific seroprevalence based on logit
# difference with all-age seroprevalence.
####################################################

pacman::p_load(dplyr, ggplot2, lubridate, tidyr, glue, nimble)
theme_set(theme_bw(base_size = 13))
theme_update(
  text = element_text(family = "Open Sans"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

# Coding "Sites" from state.abb by HHS regions
hhs <- c(
  4, 10, 9, 6, 9, 8, 1, 3, 4, 4, 9, 10, 5, 5, 7,
  7, 4, 6, 1, 3, 1, 5, 5, 4, 7, 8, 7, 9, 1, 2,
  6, 2, 4, 8, 5, 6, 10, 3, 1, 4, 8, 4, 6, 8, 1,
  4, 10, 3, 5, 8
)
hhs_lookup <- data.frame(Site = state.abb, hhs = hhs)

# Observed Serology from CDC
csv <- file.path(
  "data",
  "serological-data",
  "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
)
df <- data.table::fread(csv)

## Calculate center dates of each round of specimen collections
df_obs <- df |>
  filter(`Date Range of Specimen Collection` != "") |>
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

# Adjusted Serology (Garcia-Carreras 2023)
csv <- file.path(
  "data",
  "serological-data",
  "proportions_infected_estimates.csv"
)
df_adj <- data.table::fread(csv)
df_adj$week <- as.Date(df_adj$week)
colnames(df_adj)[3:5] <- c("adj_mean", "adj_lci", "adj_uci")

# All-age CDC "observed" data
cols <- c(
  "Rate (%) [Anti-N, All Ages Cumulative Prevalence, Rounds 1-30 only]"
)

df_allage <- df_obs |>
  select(
    Site, mid_date,
    mean = all_of(cols)
  )

adj_0_100 <- function(x) {
  # Logit cannot take 0 or 100
  x[x >= 100] <- 99.9
  x[x <= 0] <- 0.1
  return(x)
}

## Calculate logit for adjustment without over bound
df_allage_logit <- df_allage |>
  mutate(
    lmean = arm::logit(adj_0_100(mean) / 100),
  ) |>
  select(Site, mid_date, lmean)

# Age specific data
cols <- c(
  "n [Anti-N, 0-17 Years Prevalence]",
  "n [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
  "n [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
  "n [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]",
  "Rate (%) [Anti-N, 0-17 Years Prevalence]",
  "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
  "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
  "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
)

df_age <- df_obs |>
  select(
    Site, mid_date,
    all_of(cols)
  )

## Extract metric and age from long column names
df_age_long <- df_age |>
  tidyr::pivot_longer(
    -c(Site, mid_date),
    names_to = c("metric", "age"),
    names_pattern = "(.*) \\[Anti-N, (.*) Years Prevalence.*\\]"
  )

df_age_long <- df_age_long |>
  mutate(metric = case_when(
    metric == "n" ~ "n",
    metric == "Rate (%)" ~ "mean"
  )) |>
  tidyr::pivot_wider(
    names_from = "metric",
    values_from = "value"
  ) |>
  filter(mean <= 100)

## Calculate logit for adjustment without out of bound
df_age_logit <- df_age_long |>
  mutate(l_age_mean = arm::logit(adj_0_100(mean) / 100)) |>
  select(Site, mid_date, age, n, l_age_mean)

# Calculate logit spread of each age group with respect to all-age estimates
df_age_spread <- df_age_logit |>
  left_join(df_allage_logit, by = c("Site", "mid_date")) |>
  filter(!Site %in% c("DC", "PR", "ND", "US"), mid_date < ymd("2022-03-01")) |>
  mutate(diff = l_age_mean - lmean)

# Modelling spread for each age group and each state over time
## Prepare df and input for model
df_age_spread <- df_age_spread |>
  left_join(hhs_lookup, by = "Site") |>
  mutate(
    day = as.numeric(mid_date - ymd("2020-08-01")),
    site_num = as.numeric(factor(Site)),
    age_num = as.numeric(factor(age))
  )

## "Constants" for the model
hhs_const <- df_age_spread |>
  select(site_num, hhs) |>
  distinct()
hhs_const <- hhs_const$hhs

constants <- list(
  N = nrow(df_age_spread),
  NSITES = max(df_age_spread$site_num),
  NAGE = max(df_age_spread$age_num),
  NHHS = max(df_age_spread$hhs),
  site = df_age_spread$site_num,
  age = df_age_spread$age_num,
  hhs = hhs_const
)

## Data for the model
data <- list(
  x = df_age_spread$diff,
  days = df_age_spread$day / 100,
  n = df_age_spread$n
)

## Model itself
code <- nimbleCode({
  # Likelihoods
  for (i in 1:N) {
    x[i] ~ dnorm(
      alpha0[age[i], site[i]] + alpha1[age[i], site[i]] * days[i],
      sd = sigma_re[age[i]] / sqrt(n[i])
    )
  }

  # Intermediates and priors
  for (a in 1:NAGE) {
    for (j in 1:NSITES) {
      alpha0[a, j] <- (beta0[a] + beta0s[a, j])
      alpha1[a, j] <- (beta1[a] + beta1s[a, j])

      beta0s[a, j] ~ dnorm(beta0r[a, hhs[j]], sd = sigma_0r[a])
      beta1s[a, j] ~ dnorm(beta1r[a, hhs[j]], sd = sigma_1r[a])
    }

    for (k in 1:NHHS) {
      beta0r[a, k] ~ dnorm(0, sd = sigma_0[a])
      beta1r[a, k] ~ dnorm(0, sd = sigma_1[a])
    }

    beta0[a] ~ dnorm(0, sd = 1)
    beta1[a] ~ dnorm(0, sd = 1)
    sigma_0r[a] ~ dexp(10)
    sigma_1r[a] ~ dexp(10)
    sigma_0[a] ~ dexp(10)
    sigma_1[a] ~ dexp(10)
    sigma_re[a] ~ dexp(10)
  }
})

## Run model
mod <- nimbleModel(
  code = code, constants = constants,
  data = data
)
monitors <- c(
  "beta0", "beta1", "beta0r", "beta1r", "alpha0", "alpha1",
  "sigma_re", "sigma_0", "sigma_1", "sigma_0r", "sigma_1r"
)
mcmc <- buildMCMC(mod, monitors = monitors)
cmod <- compileNimble(mod)
cmcmc <- compileNimble(mcmc, project = mod)
samp <- runMCMC(cmcmc,
  nburnin = 10000, niter = 20000, nchains = 4,
  thin = 10, samplesAsCodaMCMC = TRUE, summary = TRUE
)

## Check fit
MCMCvis::MCMCsummary(samp$samples)
m <- samp$summary$all.chains[, 1]
alpha0s <- m[stringr::str_detect(names(m), "alpha0")] |>
  matrix(nrow = 4)
alpha1s <- m[stringr::str_detect(names(m), "alpha1")] |>
  matrix(nrow = 4)
df_age_spread$alpha0 <- alpha0s[as.matrix(df_age_spread[, c("age_num", "site_num")])]
df_age_spread$alpha1 <- alpha1s[as.matrix(df_age_spread[, c("age_num", "site_num")])]
df_spread_pred <- df_age_spread |>
  mutate(pred = alpha0 + alpha1 * day / 100)

ggplot(df_spread_pred) +
  geom_point(aes(x = diff, y = pred)) +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~age)

ggplot(df_spread_pred |> filter(age_num == 3)) +
  geom_point(aes(x = mid_date, y = diff), colour = "blue") +
  geom_point(aes(x = mid_date, y = pred), colour = "red") +
  facet_wrap(~Site)

# Create actual age-specific seroprevalence based on model output
## Extract the alphas and make sure they match with the correct states
## and age groups
Site_f <- unique(df_spread_pred$Site) |> sort()
age_f <- unique(df_age_spread$age) |> sort()
df_alphas <- expand.grid(age_num = 1:4, site_num = 1:49)
df_alphas$alpha0 <- alpha0s[as.matrix(df_alphas[, c("age_num", "site_num")])]
df_alphas$alpha1 <- alpha1s[as.matrix(df_alphas[, c("age_num", "site_num")])]
df_alphas$Site <- Site_f[df_alphas$site_num]
df_alphas$age <- age_f[df_alphas$age_num]

## Garcia-Carreras only goes up to 2022-01-15, appending round 30
## results from CDC as Feb 2022 data
feb2022 <- df_allage_logit |>
  filter(mid_date > ymd("2022-02-01"), mid_date < ymd("2022-03-01")) |>
  rename(state = Site, week = mid_date, adj_mean = lmean) |>
  mutate(
    adj_mean = arm::invlogit(adj_mean),
    week = mean(week)
  )
df_adj_augment <- bind_rows(df_adj, feb2022)

## Join all age data with the alphas (from the model)
## Then apply the adjustment to obtain age-specific data
comb_age_adj <- df_alphas |>
  left_join(df_adj_augment,
    by = c("Site" = "state"),
    relationship = "many-to-many"
  )
comb_age_adj <- comb_age_adj |>
  mutate(
    adj_lmean = arm::logit(adj_mean),
    day = as.numeric(week - ymd("2020-08-01")),
    fin_adj_lmean = alpha0 + alpha1 * day / 100 + adj_lmean,
    fin_adj_mean = arm::invlogit(fin_adj_lmean)
  )

ggplot(comb_age_adj) +
  geom_line(aes(x = week, y = fin_adj_mean, colour = age)) +
  facet_wrap(~Site)

# Create a special entry for HHS region 8 which is the weighted
# average of the component states trends
## ND is missing from Garcia-Carreras and we impute it with R8
df_catchment <- df |>
  select(Round, Site, `Catchment population`) |>
  filter(Round == 20) |>
  distinct() |>
  select(Site, pop = `Catchment population`)

hhs_8 <- comb_age_adj |>
  filter(Site %in% c("MT", "WY", "SD", "UT", "CO")) |>
  left_join(df_catchment, by = "Site")

hhs_8 <- hhs_8 |>
  group_by(age, week) |>
  summarise(fin_adj_mean = sum(fin_adj_mean * pop) / sum(pop))
hhs_8$Site <- "R8"

# Create a US-wide age-specific seroprevalence based on simple
# weighted average of states
uswide <- comb_age_adj |>
  left_join(df_catchment, by = "Site") |>
  group_by(age, week) |>
  summarise(fin_adj_mean = sum(fin_adj_mean * pop) / sum(pop))
uswide$Site <- "US"

# Assembling the final output and write it out
output <- bind_rows(comb_age_adj, hhs_8, uswide) |>
  select(Site, age, date = week, infected = fin_adj_mean) |>
  filter(date >= ymd("2020-08-01"))

ggplot(output) +
  geom_line(aes(x = date, y = infected, colour = age)) +
  facet_wrap(~Site)

outfile <- "./data/serological-data/seroprevalence_50states.csv"
data.table::fwrite(output, outfile)
