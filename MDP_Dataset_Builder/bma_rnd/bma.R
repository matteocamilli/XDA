suppressMessages(library(BMA))
suppressMessages(library(MASS))
suppressMessages(library(BAS))
suppressMessages(library(argparse))
suppressMessages(library(R.utils))

defaultW <- getOption('warn')
options(warn = -1)

parser <- ArgumentParser()

parser$add_argument('-s', '--sample', type='integer', default=200,
                    help='Number of observations [default 200]',
                    metavar='sample')
parser$add_argument('-v', '--vars', type='integer', default=2,
                    help='Number of variables [default 2]',
                    metavar='vars')
parser$add_argument('-m', '--method', type='character', default='MCMC',
                    help='Sampling method [default MCMC]',
                    metavar='method')

args <- parser$parse_args()

SAMPLE <- args$sample # in [200, 25600]
VARS <- args$vars # in [2, 64]
METHOD <- args$method # in {MCMC, BAS, MCMC+BAS}

df <- read.csv('req_0.csv')

bma <- function(sample, selected_method = 'MCMC') {
  reg <- bas.glm(y ~ ., data=sample,
          method=selected_method, MCMC.iterations=5000,
          betaprior=bic.prior(), family=binomial(link = "logit"),
          modelprior=uniform())
  return(reg)
}

# method="BAS+MCMC"
run_bma_timeout <- function(df, sample_size = 200, vars = 2, max_cols = 10, selected_method = 'MCMC') {
  sample <- df[1:sample_size,(max_cols-vars):max_cols]
  t0 <- Sys.time()
  bas_reg <- withTimeout(
    bma(sample, selected_method),
    timeout = 200.0, elapsed = 200.0, onTimeout = 'silent')
  time_diff <- as.numeric(Sys.time() - t0, units = "secs")
  cat(selected_method, vars, sample_size, time_diff, sep = ' ')
}

# Run without timer
run_bma <- function(df, sample_size = 200, vars = 2, max_cols = 10, selected_method = 'MCMC') {
  sample <- df[1:sample_size,(max_cols-vars):max_cols]
  t0 <- Sys.time()
  bas_reg <- bma(sample, selected_method)
  time_diff <- as.numeric(Sys.time() - t0, units = "secs")
  cat(selected_method, vars, sample_size, time_diff, sep = ' ')

  return(bas_reg)
}

# run_bma(df, SAMPLE, VARS, 64, METHOD)

reg <- run_bma(df, 5000, 9, 10, 'MCMC')

options(warn = defaultW)