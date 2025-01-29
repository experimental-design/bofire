import bofire.priors.api as priors
from bofire.data_models.priors.api import THREESIX_LENGTHSCALE_PRIOR
from bofire.plot.api import plot_prior_pdf_plotly


def test_plot_prior_pdf_plotly():
    plot_prior_pdf_plotly([priors.map(THREESIX_LENGTHSCALE_PRIOR())], lower=0, upper=10)
