import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
TOP_LEVEL = os.path.dirname(os.path.dirname(__file__))

def my_savefig(fig, filename):
    fig.savefig(f"{TOP_LEVEL}/paper/figures/{filename}.pdf",
        bbox_inches = "tight",
        format = "pdf",
        dpi = 300
    )
    fig.savefig(f"{TOP_LEVEL}/paper/figures/{filename}.png",
        bbox_inches = "tight",
        format = "png",
        dpi = 300
    )
    fig.savefig(f"{TOP_LEVEL}/paper/figures/{filename}.svg",
        bbox_inches = "tight",
        format = "svg"
    )
    plt.close(fig)

def heatmap(fig, ax, x, y, data, **kwargs):
    # Set plot defaults
    if "cmap" not in kwargs.keys():
        kwargs["cmap"] = "inferno"
    if "norm" not in kwargs.keys():
        kwargs["norm"] = matplotlib.colors.Normalize()
    if "clabel" not in kwargs.keys():
        kwargs["clabel"] = ""
    if "colorbar" not in kwargs.keys():
        kwargs["colorbar"] = True


    x_mesh, y_mesh = np.meshgrid(x, y)
    img = ax.pcolormesh(x_mesh, y_mesh, data,
        cmap = kwargs["cmap"],
        norm = kwargs["norm"],
        rasterized = True
    )
    if kwargs["colorbar"]:
        fig.colorbar(img,
            label = kwargs["clabel"]
        )
    return img

def example_model_histograms():
    data = np.load(f"{TOP_LEVEL}/data/figure_data/example_histograms.npz")

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize = (11.2, 3.2)) # (14, 4)
    fig.patch.set_facecolor("none") # Outside bg color

    # Draw data
    species = ["proton", "electron", "gamma"]
    for i in range(3):
        img = _model_histogram(fig, ax[i], data, species[i])

    # Draw main colorbar
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.7])
    fig.colorbar(img,
        label = "Counts, # per # input protons",
        cax = cbar_ax,
    )

    # Save figure
    my_savefig(fig, "example_histograms")

def _model_histogram(fig, ax, data, species):
    img = heatmap(fig, ax, data["energy_bins_mean_keV"], data["altitude_km"], data[f"{species}_counts"]/data["n_particles"],
        norm = matplotlib.colors.LogNorm(vmin = 10**-5, vmax = 10**0),
        clabel = "",
        colorbar = False,
        cmap = "bone"
    )

    # Axes
    if species == "gamma":
        ax.set_title("Photons")
    else:
        ax.set_title(f"{species.title()}s")

    ax.set_xlabel("Energy, keV")
    ax.set_xscale("log")
    ax.set_xticks(np.logspace(-2, 7, 10))

    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(1, 10)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if species in ["proton"]:
        ax.set_ylabel("Altitude, km")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 100.1, 10))

    # Misc
    ax.set_facecolor("black") # Inside bg color
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)
    ax.grid(alpha = 0.25)

    return img

def stopping_power():
    data = np.load(f"{TOP_LEVEL}/data/figure_data/stopping_power.npz")

    fig, ax = plt.subplots(1, 2, figsize = (10, 10))

    ax[0].plot(data["proton_energy_kev"], data["proton_stopping_power"],
        label = "Proton",
        color = "#f41e97"
    )
    ax[0].plot(data["alpha_energy_kev"], data["alpha_stopping_power"],
        label = "Alpha",
        color = "#8809ff",
        linestyle = "dashed"
    )
    ax[0].plot(data["electron_energy_kev"], data["electron_stopping_power"],
        label = "Electron",
        color = "#005EEB",
        linestyle = "dotted"
    )
    ax[1].plot(data["photon_energy_kev"], data["photon_mu_en"],
        label = "NIST Tables",
        color = "#f41e97"
    )
    ax[1].plot(data["photon_energy_extrapolated_kev"], data["photon_mu_en_extrapolated"],
        label = "Extrapolation",
        color = "#888888",
        linestyle = "dashed"
    )


    # X-axis
    ax[1].set_xlim(10.0**0, 10.0**7)

    # Y-axis
    ax[0].set_ylabel("Stopping Power, keV m² kg⁻¹")
    ax[1].set_ylabel("μₑₙ, m² kg⁻¹")

    #ax.set_ylim(0, 100)
    #ax.set_yticks(range(0, 101, 10))

    # Misc/box
    for i, _ in enumerate(ax):
        ax[i].set_xlabel("Energy, keV")

        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

        ax[i].legend()
        ax[i].set_box_aspect(1/1.5)
        ax[i].set_facecolor("none") # Inside bg color
        ax[i].minorticks_on()
        ax[i].grid(
            which = "major",
            alpha = 0.5
        )
        ax[i].grid(
            which = "minor",
            alpha = 0.1
        ) 
        ax[i].set_axisbelow(True)

    my_savefig(fig, "stopping_power")

def case_studies():
    data = np.load(f"{TOP_LEVEL}/data/figure_data/example_armas_data.npz")

    fig, ax = plt.subplots(1, 4, figsize = (12, 10))
    fig.patch.set_facecolor("none") # Outside bg color
    [_case_study(ax[i], data, i) for i in range(4)]

    my_savefig(fig, "case_studies")

def _case_study(ax, data, i):
    # Load data
    data_names = ["agree", "nairas_under_1", "nairas_under_2", "excess"]
    unfortunately_dates_must_be_hardcoded = ["2015-12-19", "2019-09-14", "2019-09-05", "2017-10-14"]

    name = data_names[i]
    altitude = data[f"{name}_altitude"]
    nairas = data[f"{name}_nairas"]
    armas = data[f"{name}_armas"]
    glyphs_altitude = data[f"{name}_glyphs_altitude"]
    glyphs_dose = data[f"{name}_glyphs_doserate"]
    glyphs_dose_no_extrap = data[f"{name}_glyphs_doserate_no_extrap"]

    # Plot data
    ax.plot(glyphs_dose, glyphs_altitude,
        label = "SEPIIDA (this work)",
        color = "#ff71a3",
    )
    ax.scatter(armas, altitude,
        label = "ARMAS",
        marker = ".",
        color = "black",
        s = 50
    )
    ax.errorbar(armas, altitude,
        linestyle = "None",
        xerr = 0.2 * armas,
        color = "black",
        linewidth = .5,
        alpha = 0.35
    )
    ax.scatter(nairas, altitude,
        label = "NAIRAS",
        marker = "x",
        color = "#1dd1db",
        s = 30,
        linewidth = 1
    )

    # Panel labels
    letters = ['a', 'b', 'c', 'd']
    ax.annotate(f"{letters[i]}", (0.2, 19),
        size = 18,
        horizontalalignment = "left",
        verticalalignment = "center",
    )

    # Axis parameters
    if i == 0: ax.legend(loc = "lower right")
    ax.set_title(unfortunately_dates_must_be_hardcoded[i])
    

    ax.set_xlabel("Doserate, μGy/hr")
    ax.set_xlim(0, 6)
    ax.set_xticks(np.arange(0, 6.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    if i == 0: ax.set_ylabel("Altitude, km")
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 20.1, 2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
    
    ax.set_box_aspect(2)
    ax.grid(
        alpha = 0.5
    )
    ax.set_facecolor("none") # Inside bg color
    ax.set_axisbelow(True)

def armas_residuals():
    fig, ax = plt.subplots(1, 2,
        figsize=(9, 9),
        height_ratios = [1]
    )
    fig.patch.set_facecolor("none") # Outside bg color

    _residual_histogram(ax[0], "SEPIIDA")
    _residual_histogram(ax[1], "NAIRAS")

    # Save
    fig.tight_layout()
    my_savefig(fig, "armas_residuals")

def _residual_histogram(ax, compare_against):
    data = np.load(f"{TOP_LEVEL}/data/figure_data/conjunction_data.npz")
    to_plot = np.concatenate((data[f"{compare_against.lower()}_residuals"], [data[f"{compare_against.lower()}_residuals"][-1]]), axis = 0)
    ax.step(data["residual_bin_edges"], to_plot,
        color = "black",
        where = "post",
    )
    ax.fill_between(data["residual_bin_edges"], to_plot,
        step = "post",
        color = "black",
        alpha = 0.1
    )

    # X-axis
    ax.set_xlabel(f"ARMAS - {compare_against}, μGy/hr")
    ax.set_xlim(-5, 5)
    ax.set_xticks(np.arange(-5, 5.1, 1))

    # Y-axis
    def yticks(x, pos):
        return f"{x/1e3:n}k"
    ax.set_ylim(0, 90e3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yticks))


    if compare_against == "SEPIIDA": ax.set_ylabel("Datapoints")
    if compare_against == "NAIRAS":
        for label in ax.yaxis.get_ticklabels():
            label.set_visible(False)


    # Annotation
    ax.annotate("Model\noverestimates", (-3, 55000),
        horizontalalignment = "center",
        verticalalignment = "center",
    )
    ax.annotate("Model\nunderestimates", (3, 55000),
        horizontalalignment = "center",
        verticalalignment = "center",
    )

    # Misc/box
    ax.set_axisbelow(True)
    ax.set_box_aspect(1/1.5)
    ax.set_facecolor("none") # Inside bg color
    ax.minorticks_on()
    ax.grid(alpha = 0.4)

def elfin_example_data():
    data = np.load(f"{TOP_LEVEL}/data/figure_data/elfin_event.npz")

    fig, ax = plt.subplots(figsize = (4.48, 3.64))
    heatmap(fig, ax, data["pitch_angles"], data["energy_bins_mean_kev"], data["n_flux"],
        norm = matplotlib.colors.LogNorm(),
        cmap = "bone",
        clabel = "Electron Flux, # cm⁻² s⁻¹ str⁻¹ MeV ⁻¹"
    )

    # Title
    ax.set_title("ELFIN-B 2022-02-04 15:17:32.015")

    # X-axis
    ax.set_xlabel("Pitch Angle, deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    # Y-axis
    ax.set_ylabel("Energy, keV")
    ax.set_yscale("log")
    ax.set_ylim(50, 7000)

    # Misc/box
    ax.set_box_aspect(1)
    ax.set_facecolor("black") # Inside bg color
    ax.set_axisbelow(True)

    my_savefig(fig, "elfin_example_data")

def electron_dose():
    fig, ax = plt.subplots(1, 2,
        figsize=(10, 5),
        width_ratios = [2, 1],
        height_ratios = [1]
    )
    fig.patch.set_facecolor("none") # Outside bg color
    _electron_input_spectrum(ax[0])
    _electron_doserate(ax[1])

    my_savefig(fig, "electron_spectrum")

def _electron_input_spectrum(ax):
    data = np.load(f"{TOP_LEVEL}/data/figure_data/elfin_electron_doserate.npz")

    ax.plot(data["energy_labels"], data["top_energy_spectrum"],
        marker = "o",
        color = "#f41e97"
    )
    ax.plot(data["energy_labels"], data["percentile_98_energy_spectrum"],
        marker = "o",
        color = "#8809ff",
        linestyle = "--"
    )
    ax.plot(data["energy_labels"], data["percentile_90_energy_spectrum"],
        marker = "o",
        color = "#005EEB",
        linestyle = "-."
    )

    # Title
    ax.set_title("(a)")

    # X-axis
    ax.set_xlabel("Energy, keV")
    ax.set_xscale("log")
    ax.set_xlim(0.9 * data["energy_labels"][0], 1.1 * data["energy_labels"][-1])
    #ax.set_xticks(10.0 ** np.arange(-2, 4.1, 1))

    # Y-axis
    ax.set_ylabel("Precipitating Electron Flux, # m⁻² s⁻¹")
    ax.set_yscale("log", nonpositive = "mask")
    ax.set_ylim(10**5, 10**10)

    # Misc/box
    ax.grid(which = "major")
    ax.grid(which = "minor",
        alpha = 0.3
    )
    ax.minorticks_on()
    ax.set_facecolor("none") # Inside bg color
    ax.set_axisbelow(True)

def _electron_doserate(ax):
    data = np.load(f"{TOP_LEVEL}/data/figure_data/elfin_electron_doserate.npz")

    ax.set_title("(b)")
    ax.plot(data["top_doserate_uGy/hr"], data["altitude_km"],
        label = "99.9th Percentile",
        color = "#f41e97"
    )
    ax.plot(data["percentile_98_doserate_uGy/hr"], data["altitude_km"],
        label = "98th Percentile",
        color = "#8809ff",
        linestyle = "--"
    )
    ax.plot(data["percentile_90_doserate_uGy/hr"], data["altitude_km"],
        label = "90th Percentile",
        color = "#005EEB",
        linestyle = "-."
    )

    # X-axis
    ax.set_xlabel("Doserate, μGy/hr")
    ax.set_xscale("log")
    ax.set_xlim(10.0**-4, 10.0**8)
    ax.set_xticks(10.0 ** np.arange(-4, 8.1, 2))

    # Y-axis
    ax.set_ylabel("Altitude, km")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))

    # Misc/box
    ax.legend(
        loc='upper center',
        bbox_to_anchor = (-1, -0.15),
        ncol = 3
    )
    ax.set_box_aspect(2)
    ax.set_facecolor("none") # Inside bg color
    ax.minorticks_on()
    ax.grid()
    ax.set_axisbelow(True)

def gcr_dose():
    fig, ax = plt.subplots(1, 2,
        figsize=(10, 5),
        width_ratios = [2, 1],
        height_ratios = [1]
    )
    fig.patch.set_facecolor("none") # Outside bg color
    _gcr_spectrum(ax[0])
    _gcr_dose(ax[1])

    my_savefig(fig, "gcr_spectrum")

def _gcr_spectrum(ax):
    data = np.load(f"{TOP_LEVEL}/data/figure_data/gcr_spectrum.npz")

    ax.plot(data["proton_energy_bins_mean_GeV"], data["proton_flux"],
        label = "Protons",
        color = "#ff8fb7",
        marker = "o"
    )
    ax.plot(data["alpha_energy_bins_mean_GeV"], data["alpha_flux"],
        label = "Alpha Particles",
        color = "#1dd1db",
        marker = "o",
        linestyle = "dashed"
    )

    # Title
    ax.set_title("AMS-02 2014-02-21")

    # X-axis
    ax.set_xlabel("Energy, GeV")
    ax.set_xscale("log")

    # Y-axis
    ax.set_ylabel("Flux, # s⁻¹ m⁻² str⁻¹ GeV⁻¹")
    ax.set_yscale("log")

    # Misc/box
    ax.legend(loc = "upper right")
    ax.grid(which = "major")
    ax.grid(which = "minor",
        alpha = 0.3
    )

    ax.set_axisbelow(True)
    ax.set_facecolor("none") # Inside bg color

def _gcr_dose(ax):
    data = np.load(f"{TOP_LEVEL}/data/figure_data/gcr_spectrum.npz")

    ax.plot(data["doserate_uGy/hr"], data["altitude_km"],
        color = "black"
    )

    # Title
    ax.set_title("AMS-02 2014-02-21")

    # X-axis
    ax.set_xlabel("Doserate, μGy/hr")
    ax.set_xscale("log")
    ax.set_xlim(10**-1, 10**1)

    # Y-axis
    ax.set_ylabel("Altitude, km")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))

    # Misc/box
    ax.grid(which = "major")
    ax.grid(which = "minor",
        alpha = 0.3
    )

    ax.set_box_aspect(2)
    ax.set_facecolor("none") # Inside bg color
    ax.set_axisbelow(True)

def excess_doserate_comparison():
    data = np.load(f"{TOP_LEVEL}/data/figure_data/conjunction_data.npz")
    electron_data = np.load(f"{TOP_LEVEL}/data/figure_data/elfin_electron_doserate.npz")

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4.55))

    heatmap(fig, ax, data["excess_dose_bin_edges"], data["altitude_bin_edges"], data["excess_dose_histogram"],
        cmap = "plasma",
        norm = matplotlib.colors.LogNorm(),
        clabel = "Number of Datapoints"
    )
    ax.plot(electron_data["top_doserate_uGy/hr"], electron_data["altitude_km"],
        label = "99.9th Percentile",
        color = "#f41e97"
    )
    ax.plot(electron_data["percentile_98_doserate_uGy/hr"], electron_data["altitude_km"],
        label = "98th Percentile",
        color = "#8809ff",
        linestyle = "--"
    )
    ax.plot(electron_data["percentile_90_doserate_uGy/hr"], electron_data["altitude_km"],
        label = "90th Percentile",
        color = "#005EEB",
        linestyle = "-."
    )
    # X-axis
    ax.set_xlabel("Excess Doserate, μGy/hr")
    ax.set_xscale("log")
    ax.set_xticks(np.logspace(-5, 1, 7))
    ax.set_xlim(10**-5, 10**2)

    # Y-axis
    ax.set_ylabel("Altitude, km")
    ax.set_ylim(6, 24)
    ax.set_yticks(range(6, 25, 2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    # Misc
    ax.legend(
        loc = "upper left"
    )
    ax.grid(
        which = "major",
        alpha = 0.4
    )
    ax.grid(
        which = "minor",
        alpha = 0.1
    )
    ax.set_box_aspect(0.8)
    ax.set_facecolor("none") # Inside bg color
    ax.set_axisbelow(True)

    # Save figure
    my_savefig(fig, "excess_dose_comparison")

stopping_power()
example_model_histograms()
gcr_dose()
case_studies()
armas_residuals()
elfin_example_data()
electron_dose()
excess_doserate_comparison()