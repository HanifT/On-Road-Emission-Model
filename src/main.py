# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/On_Road_Emission_Model/src')
from reading_input import load_csv_as_dataframe
import utilities
# %%
# Reading the file
input_df = load_csv_as_dataframe('/Users/haniftayarani/On_Road_Emission_Model/Data/Input_Data.csv')

# %% Q1
"""
1a. How do electric heavy-duty truck (MHDT, HHDT) and bus (OBUS, SBUS) populations change over time?
"""
q1_instance = utilities.Q1(input_df)
q1_instance.filter_by_fuel_type()
q1_instance.filter_by_multiple_choices()
q1_instance.group_and_sum_column('Population')
q1_instance.plot_stacked_bar_with_line("Population")
# %%
"""
1b. Which county is projected to have the largest electric heavy-duty truck population by 2040?
"""
q1_instance = utilities.Q1(input_df)
q1_instance.filter_by_fuel_type()
q1_instance.filter_by_multiple_choices()
q1_instance.plot_top_regions_population(2040)

# %% Q2
"""
2. CARB's Clean Truck Check Program ensures that proper maintenance of vehicle emissions
aftertreatment systems. The impacts of this program were not reflected in EMFAC2021, since
the regulation was adopted after the release of the model.
Please use the information provided in the table below to estimate the impact of this
regulation. Please note that these reduction factors depend on calendar year, which
represents different phases of the program, and NOx emissions reductions only apply to
vehicles with EMFAC model years 2014 or newer that are equipped with on-board diagnostics
systems. Please calculate NOx and PM2.5 total exhaust emissions (denoted as TOTEX) before
and after this regulation by calendar year.
"""
q2_instance = utilities.Q2(input_df)
baseline_emissions = q2_instance.selection()
q2_instance.baseline_emissions = baseline_emissions.apply(q2_instance.apply_reduction, axis=1)
q2_instance.plot_emission_reduction()

# %% Q3
"""
Emission rates can be characterized as grams of emissions per mile traveled (g/mile). Using
the data provided, please calculate the statewide fleet average running exhaust NOx
emission rate of heavy heavy-duty trucks (HHDT) from 2020 to 2040? Which county has the
highest NOx emission rate in 2035? Note that running exhaust NOx is denoted as
NOx_RUNEX. Hint: the fleet averages should be weighted by VMT.
"""
q3_instance = utilities.Q3(input_df, '/Users/haniftayarani/Library/CloudStorage/Box-Box/GOOD Model/cb_2018_us_county_5m')
q3_instance.calculate_nox_emission_rates()
q3_instance.plot_fleet_average_nox()
q3_instance.plot_nox_heat_map(2035)
q3_instance.plot_yearly_average_nox_heat_map(2035)

# %% Q4

q4_instance = utilities.Q4(input_df)
q4_instance.get_summary_statistics()
q4_instance.plot_vmt_inconsistencies()
q4_instance.analyze_fuel_inconsistencies()
q4_instance.plot_energy_inconsistencies_heatmap()
q4_instance.run_analysis()
q4_instance.analyze_gas_data()

