import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact

# Bayesian Inference Function
def estimate_aav_parameters(observed_incidence, observed_prevalence, observed_mortality, observed_relapse_rate, observed_treatment_effect):
    with pm.Model() as model:
        # Priors for parameters
        incidence = pm.Uniform("incidence", lower=0.1, upper=10)
        prevalence = pm.Uniform("prevalence", lower=1, upper=50)
        mortality = pm.Uniform("mortality", lower=0.01, upper=0.3)
        relapse_rate = pm.Uniform("relapse_rate", lower=0.01, upper=0.5)
        treatment_effect = pm.Uniform("treatment_effect", lower=0.1, upper=0.9)

        # Likelihood based on observed data
        observed_inc = pm.Normal("observed_incidence", mu=incidence, sigma=1, observed=observed_incidence)
        observed_prev = pm.Normal("observed_prevalence", mu=prevalence, sigma=5, observed=observed_prevalence)
        observed_mort = pm.Normal("observed_mortality", mu=mortality, sigma=0.05, observed=observed_mortality)
        observed_relapse = pm.Normal("observed_relapse_rate", mu=relapse_rate, sigma=0.1, observed=observed_relapse_rate)
        observed_treat_eff = pm.Normal("observed_treatment_effect", mu=treatment_effect, sigma=0.1, observed=observed_treatment_effect)

        # Sampling from the posterior
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # Display summary of posterior distributions
    summary = pm.summary(trace)
    print("\nPosterior Summary:")
    display(summary)

    # Plot posterior distributions
    fig, axes = plt.subplots(5, 1, figsize=(8, 20))
    pm.plot_posterior(
        trace,
        var_names=["incidence", "prevalence", "mortality", "relapse_rate", "treatment_effect"],
        ax=axes
    )
    plt.tight_layout()
    plt.show()

# Interactive Widgets for Input
observed_incidence_widget = widgets.FloatSlider(value=2.5, min=0.1, max=20.0, step=0.1, description="Incidence")
observed_prevalence_widget = widgets.FloatSlider(value=30.0, min=0.1, max=100.0, step=0.1, description="Prevalence")
observed_mortality_widget = widgets.FloatSlider(value=0.1, min=0.01, max=0.3, step=0.01, description="Mortality")
observed_relapse_rate_widget = widgets.FloatSlider(value=0.2, min=0.01, max=0.5, step=0.01, description="Relapse Rate")
observed_treatment_effect_widget = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.1, description="Treatment Effect")

# Interact Function
interact(
    estimate_aav_parameters,
    observed_incidence=observed_incidence_widget,
    observed_prevalence=observed_prevalence_widget,
    observed_mortality=observed_mortality_widget,
    observed_relapse_rate=observed_relapse_rate_widget,
    observed_treatment_effect=observed_treatment_effect_widget
)