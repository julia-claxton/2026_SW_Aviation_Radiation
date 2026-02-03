This README file was generated on 2022-08-26 by INFN group @SSDC, last update on 2025-06-12


GENERAL INFORMATION

Title of Dataset: AMS time dependent fluxes
Author: AMS Collaboration


DATA & FILE OVERVIEW

File List:
README: general info about the dataset
p_AMS_PRL2021_daily.csv : data table for AMS daily proton fluxes collected from May 20, 2011 to October 29, 2019 in the rigidity interval from 1 to 100 GV
He_AMS_PRL2022_daily.csv : data table for AMS daily Helium fluxes collected from May 20, 2011 to October 29, 2019 in the rigidity interval from 1.71 to 100 GV
e-_AMS_PRL2023_daily.csv : data table for AMS daily electrons fluxes collected from May 20, 2011 to November 02, 2021 in the rigidity interval from 1 to 41.9 GV
e+_AMS_PRL2023_daily.csv : data table for AMS daily positrons fluxes collected from May 20, 2011 to November 02, 2021 in the rigidity interval from 1 to 41.9 GV
D-He3-He4_AMS_PRL2024_4BR.csv : data tables for AMS Deuteron, Helium-3, and Helium-4 fluxes, and the D/He4 and He3/He4 flux ratios measured in 33 time periods of four Bartels
rotations (108 days) each from May 30, 2011 to April 09, 2021 in the rigidity interval from 1.92 to 21.1 GV
pbar_AMS_PRL2025_BR.csv : data table for AMS antiproton fluxes for each Bartels rotation (BR:27 days) measured from May 2011 to June 2022 (139 BR) in the rigidity interval from 1.00 to 41.9 GV
He_AMS_PRL2025_BR.csv : data table for AMS time dependent Helium fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
Li-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Lithium fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
Be-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Beryllium fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
B-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Boron fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
C-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Carbon fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
N-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Nitrogen fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
O-He_AMS_PRL2025_BR.csv : data table for AMS time dependent Oxygen fluxes measured in 147 Bartels rotations from May 15, 2011 to November 25, 2022 in the rigidity range from 1.92 to 60.3 GV
6Li-7Li_AMS_PRL2025_4BR.csv : data tables for AMS Lithium-6, Lithium-7 fluxes and Li6/Li7 flux ratios measured in 42 time periods of four Bartels rotations (108 days) each from May 2011 to April 2021 in the rigidity interval from 1.92 to 24.7 GV


Unfortunately, due to limitations in the current version of the CRDB (v3.2), this dataset is only available for direct download and not for query nor visualisation. You can download it from both the CRDB homepage (https://tools.ssdc.asi.it/CosmicRays/) and from the CRDB Dataset Reference Table (https://tools.ssdc.asi.it/CosmicRays/cosmicdataset.jsp).
More data from time-dependent flux measurements in space are available accessing the CRDB via usual queries, and can be both visualised and downloaded.

Dataset last uploaded on 2025-06-10


DATA-SPECIFIC INFORMATION FOR: p_AMS_PRL2021_daily.csv

Number of variables: 6

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
date: days are defined as UTC time 00:00:00 - 23:59:59
rigidity_min: minimum of the rigidity range in GV
rigidity_max: maximum of the rigidity range in GV
proton_flux: daily proton flux in m^-2sr^-1s^-1GV^-1
proton_flux_errror_statistical: statistical error in m^-2sr^-1s^-1GV^-1
proton_flux_error_timedependent: time-dependent systematic errors (from the trigger efficiency and the reconstruction efficiencies) in m^-2sr^-1s^-1GV^-1
proton_flux_error_systematic_total: total systematic error (with contributions from the time-dependent systematic error, the background evaluation, the geomagnetic cutoff, the rigidity resolution function and the absolute rigidity scale) in m^-2sr^-1s^-1GV^-1


DATA-SPECIFIC INFORMATION FOR: He_AMS_PRL2022_daily.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
date: days are defined as UTC time 00:00:00 - 23:59:59
rigidity_min: minimum of the rigidity range in GV
rigidity_max: maximum of the rigidity range in GV
helium_flux: daily helium flux in m^-2sr^-1s^-1GV^-1
helium_flux_error_statistical: statistical error in m^-2sr^-1s^-1GV^-1
helium_flux_error_timedependent: time-dependent systematic errors (from the trigger efficiency, the reconstruction efficiencies, and the unfolding) in m^-2sr^-1s^-1GV^-1
helium_flux_error_systematic_total: total systematic error (with contributions from the time- dependent systematic error, the background evaluation, the geomagnetic cutoff, the acceptance calculation, the rigidity resolution function, and the absolute rigidity scale) in m^-2sr^-1s^-1GV^-1
helium_to_proton_ratio: daily helium to proton flux ratios at the top of AMS
helium_to_proton_ratio_error_statistical: sum in quadrature of the relative statistical errors of the fluxes multiplied by the ratio
helium_to_proton_ratio_error_timedependent: the time-dependent systematic errors for the ratio are the sum in quadrature of the relative time-dependent systematic errors of the fluxes multiplied by the ratio
helium_to_proton_ratio_error_systematic_total: The contributions of the individual sources to the systematic error are added in quadrature to arrive at the total systematic uncertainty on the ratio. The time-independent systematic errors for the ratio are the sum in quadrature of the relative time-independent systematic errors of the fluxes multiplied by the ratio, taking into account the correlations in the systematic errors from the unfolding and the absolute rigidity scale between the helium and proton fluxes.


DATA-SPECIFIC INFORMATION FOR: e-_AMS_PRL2023_daily.csv

Number of variables: 7

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
date: days are defined as UTC time 00:00:00 - 23:59:59
rigidity_min: minimum of the rigidity range in GV
rigidity_max: maximum of the rigidity range in GV
electron_flux: daily electron flux in m^-2sr^-1s^-1GV^-1
electron_flux_error_statistical: statistical error in m^-2sr^-1s^-1GV^-1,
electron_flux_error_timedependent: time-dependent systematic errors (with contributions from the template definition, the charge confusion, the trigger efficiency, the reconstruction efficiencies, and the unfolding) in m^-2sr^-1s^-1GV^-1,
electron_flux_error_systematic_total: total systematic error (with contributions from the time-dependent systematic error, the geomagnetic cutoff, the rigidity resolution function, and the absolute rigidity scale) in m^-2sr^-1s^-1GV^-1


DATA-SPECIFIC INFORMATION FOR: e+_AMS_PRL2023_daily.csv

Number of variables: 7

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
date: days are defined as UTC time 00:00:00 - 23:59:59
rigidity_min: minimum of the rigidity range in GV
rigidity_max: maximum of the rigidity range in GV
positron_flux: daily electron flux in m^-2sr^-1s^-1GV^-1
positron_flux_error_statistical: statistical error in m^-2sr^-1s^-1GV^-1,
positron_flux_error_timedependent: time-dependent systematic errors (with contributions from the template definition, the charge confusion, the trigger efficiency, the reconstruction efficiencies, and the unfolding) in m^-2sr^-1s^-1GV^-1,
positron_flux_error_systematic_total: total systematic error (with contributions from the time-dependent systematic error, the geomagnetic cutoff, the rigidity resolution function, and the absolute rigidity scale) in m^-2sr^-1s^-1GV^-1


DATA-SPECIFIC INFORMATION FOR: D-3He-4He_AMS_PRL2024_4BR.csv

Number of variables: 24

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
begin date YYYY-MM-DD: first day of the four Bartels rotations time period (days are defined as UTC time 00:00:00 - 23:59:59)
end date YYYY-MM-DD: last day of the four Bartels rotation time periods (days are defined as UTC time 00:00:00 - 23:59:59)
rigidity_min: minimum of the rigidity range in GV
rigidity_max: maximum of the rigidity range in GV
D_flux : deuteron flux in m^-2sr^-1s^-1GV^-1
D_flux_error_statistical: statistical error in m^-2sr^-1s^-1GV^-1
D_flux_error_timedependent : time dependent systematic error (due to the variations of trigger and reconstruction efficiencies) in m^-2sr^-1s^-1GV^-1
D_flux_error_systematic , : time independent systematic error (due to uncertainties in the rigidity and velocity resolution functions, unfolding procedure, and background subtraction, and also trigger efficiency, geomagnetic cutoff, and the effective acceptance) in m^-2sr^-1s^-1GV^-1
3He_flux : helium-3 flux in m^-2sr^-1s^-1GV^-1
3He_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1,
3He_flux_error_timedependent : time dependent systematic error in m^-2sr^-1s^-1GV^-1
3He_flux_error_systematic : time independent systematic error in m^-2sr^-1s^-1GV^-1
4He_flux : helium-4 flux in m^-2sr^-1s^-1GV^-1
4He_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1,
4He_flux_error_timedependent : time dependent systematic error in m^-2sr^-1s^-1GV^-1,
4He_flux_error_systematic : time independent systematic error in m^-2sr^-1s^-1GV^-1,
D_to_4He_ratio : flux ratio between deuterium and helium-3 fluxes
D_to_4He_ratio_error_statistical : statistical error of the ratio
D_to_4He_ratio_error_timedependent : time dependent systematic error of the ratio
D_to_4He_ratio_error_systematic : time independent systematic error of the ratio
3He_to_4He_ratio : flux ratio between Helium-3 and helium-4 fluxes
3He_to_4He_ratio_error_statistical : statistical error of the ratio
3He_to_4He_ratio_error_timedependent : time dependent systematic error of the ratio
3He_to_4He_ratio_error_systematic : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: pbar_AMS_PRL2025_BR.csv

Number of variables: 7

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min GV : minimum of the rigidity range in GV
rigidity_max GV : maximum of the rigidity range in GV
antiproton_flux : antiproton flux for each Bartels rotation in m^-2sr^-1s^-1GV^-1
antiproton_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1
antiproton_flux_error_systematic_timedependent : time dependent systematic error (including contributions from background subtraction, trigger efficiency, and unfolding) in m^-2sr^-1s^-1GV^-1,
antiproton_flux_error_systematic_total : total systematic error (with contribution from the time-dependent systematic error, the geomagnetic cutoff, the effective acceptance, the unfolding, and the rigidity scale) in m^-2sr^-1s^-1GV^-1


DATA-SPECIFIC INFORMATION FOR: He_AMS_PRL2025_BR.csv

Number of variables: 7

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
helium_flux : Helium flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
helium_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1
helium_flux_error_time_dependent : time dependent systematic error in m^-2sr^-1s^-1GV^-1,
helium_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1


DATA-SPECIFIC INFORMATION FOR: Li-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
lithium_flux: Lithium flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
lithium_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1
lithium_flux_error_time_dependent : time dependent systematic error in m^-2sr^-1s^-1GV^-1
lithium_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1
lithium_to_helium_ratio : flux ratio between Lithium and Helium fluxes per Bartels rotations
lithium_to_helium_ratio_error_statistical : statistical error of the ratio
lithium_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
lithium_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: Be-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
beryllium_flux: Beryllium flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
beryllium_flux_error_statistical : ststistical error in m^-2sr^-1s^-1GV^-1
beryllium_flux_error_time_dependent : time-dependent systematic error in m^-2sr^-1s^-1GV^-1,
beryllium_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1,
beryllium_to_helium_ratio : flux ratio between Beryllium and Helium fluxes per Bartels rotations
beryllium_to_helium_ratio_error_statistical : statistical error of the ratio
beryllium_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
beryllium_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: B-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
boron_flux : Boron flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
boron_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1,
boron_flux_error_time_dependent : time-dependent systematic error in m^-2sr^-1s^-1GV^-1,
boron_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1
boron_to_helium_ratio : flux ratio between Boron and Helium fluxes per Bartels rotations
boron_to_helium_ratio_error_statistical : statistical error of the ratio
boron_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
boron_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: C-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
carbon_flux : Carbon flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
carbon_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1,
carbon_flux_error_time_dependent : time-dependent systematic error in m^-2sr^-1s^-1GV^-1
carbon_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1
carbon_to_helium_ratio : flux ratio between Carbon and Helium fluxes per Bartels rotations
carbon_to_helium_ratio_error_statistical : statistical error of the ratio
carbon_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
carbon_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: N-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
nitrogen_flux : Nitrogen flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
nitrogen_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1,
nitrogen_flux_error_time_dependent : time-dependent systematic error in m^-2sr^-1s^-1GV^-1
nitrogen_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1
nitrogen_to_helium_ratio : flux ratio between Nitrogen and Helium fluxes per Bartels rotations
nitrogen_to_helium_ratio_error_statistical : statistical error of the ratio
nitrogen_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
nitrogen_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: O-He_AMS_PRL2025_BR.csv

Number of variables: 11

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
bartels_rotation_number : Bartels solar rotation number (27-day intervals counted from Feb 8, 1832)
rigidity_min_GV : minimum of the rigidity range in GV
rigidity_max_GV : maximum of the rigidity range in GV
oxygen_flux Oxygen flux per Bartels rotation in m^-2sr^-1s^-1GV^-1
oxygen_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1
oxygen_flux_error_time_dependent : time-dependent systematic error in m^-2sr^-1s^-1GV^-1,
oxygen_flux_error_time_independent : time independent systematic error in m^-2sr^-1s^-1GV^-1
oxygen_to_helium_ratio : flux ratio between Oxygen and Helium fluxes per Bartels rotations
oxygen_to_helium_ratio_error_statistical : statistical error of the ratio
oxygen_to_helium_ratio_error_time_dependent : time-dependent systematic error of the ratio
oxygen_to_helium_ratio_error_time_independent : time independent systematic error of the ratio


DATA-SPECIFIC INFORMATION FOR: 6Li-7Li_AMS_PRL2025_4BR.csv

Number of variables: 15

Variable List: <list variable name(s), description(s), unit(s) and value labels as appropriate for each>
begin date YYYY-MM-DD : first day of the four Bartels rotations time period (days are defined as UTC time 00:00:00 - 23:59:59)
end date YYYY-MM-DD : last day of the four Bartels rotation time periods (days are defined as UTC time 00:00:00 - 23:59:59)
rigidity_min GV : minimum of the rigidity range in GV

rigidity_max GV : maximum of the rigidity range in GV
6Li_flux Lithium-6 flux in m^-2sr^-1s^-1GV^-1
6Li_flux_error_statistical : statistical error in m^-2sr^-1s^-1GV^-1
6Li_flux_error_timedependent : time dependent systematic error in m^-2sr^-1s^-1GV^-1
6Li_flux_error_total_systematic : time independent systematic error in m^-2sr^-1s^-1GV^-1
7Li_flux : Lithium-7 flux in m^-2sr^-1s^-1GV^-1
7Li_flux_error_statistical: statistical error in m^-2sr^-1s^-1GV^-1
7Li_flux_error_timedependent: time dependent systematic error in m^-2sr^-1s^-1GV^-1
7Li_flux_error_total_systematic: total systematic error in m^-2sr^-1s^-1GV^-1
7Li_to_6Li_ratio : flux ratio between Lithium-6 and Lithium-7 fluxes
7Li_to_6Li_ratio_error_statistical : statistical error of the ratio
7Li_to_6Li_ratio_error_total_systematic: total systematic error of the ratio (note that time dependent systematic error is zero for the 7Li/6Li ratio )



