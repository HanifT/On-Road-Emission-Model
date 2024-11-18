import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import ipywidgets as widgets
from IPython.display import display

# %% Q1


class Q1():

    def __init__(self, df):
        self.df = df

    def filter_by_fuel_type(self):
        if 'Fuel' not in self.df.columns:
            print("Error: The DataFrame does not have a 'Fuel' column.")
            return None

        unique_fuels = self.df['Fuel'].unique()

        options_str = "Please select the number of the fuel type you want:\n"
        for i, fuel in enumerate(unique_fuels, 1):
            options_str += f"{i}. {fuel}\n"

        while True:
            try:
                choice = int(input(options_str))
                if 1 <= choice <= len(unique_fuels):
                    selected_fuel = unique_fuels[choice - 1]
                    print(f"You selected: {selected_fuel}")
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.filtered_df = self.df[self.df['Fuel'] == selected_fuel]
        return self.filtered_df

    def filter_by_multiple_choices(self):
        column_name = "Vehicle Category"
        if column_name not in self.filtered_df.columns:
            print(f"Error: The DataFrame does not have a '{column_name}' column.")
            return None

        self.unique_values = self.filtered_df[column_name].unique()

        options_str = f"Please select your {column_name}(s) for analysis (separated by commas):\n"
        for i, value in enumerate(self.unique_values, 1):
            options_str += f"{i}. {value}\n"

        while True:
            try:
                selected_indices = input(options_str)
                selected_indices = [int(x.strip()) for x in selected_indices.split(",")]
                if all(1 <= idx <= len(self.unique_values) for idx in selected_indices):
                    self.selected_values = [self.unique_values[idx - 1] for idx in selected_indices]
                    print(f"You selected: {self.selected_values}")
                    break
                else:
                    print("Invalid choice(s). Please enter numbers from the list.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")

        self.filtered_df = self.filtered_df[self.filtered_df[column_name].isin(self.selected_values)]
        return self.filtered_df

    def group_and_sum_column(self, sum_column):
        if 'Vehicle Category' not in self.filtered_df.columns or 'Calendar Year' not in self.filtered_df.columns:
            print("Error: The DataFrame must contain 'Vehicle Category' and 'Calendar Year' columns.")
            return None

        if sum_column not in self.filtered_df.columns:
            print(f"Error: The DataFrame does not contain the column '{sum_column}' to sum.")
            return None

        self.grouped_df = self.filtered_df.groupby(['Vehicle Category', 'Calendar Year'])[sum_column].sum().reset_index()

        return self.grouped_df

    def plot_stacked_bar_with_line(self, sum_column):
        self.grouped_df = self.filtered_df.groupby(['Calendar Year', 'Vehicle Category'])[sum_column].sum().unstack().fillna(0)

        total_sum_per_year = self.grouped_df.sum(axis=1)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        self.grouped_df.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')

        ax1.set_xlabel("Calendar Year", fontsize=14)
        ax1.set_ylabel(f"Sum of {sum_column} by Vehicle Category", fontsize=14)
        ax1.set_title(f"{sum_column} by Vehicle Category with Total Sum Line", fontsize=14)
        ax1.tick_params(axis='x', labelsize=12)  # Increase x-axis tick label size
        ax1.tick_params(axis='y', labelsize=12)  # Increase y-axis tick label size

        ax1.legend(title="Vehicle Category", fontsize=12, title_fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_top_regions_population(self, year):
        df_year = self.filtered_df[self.filtered_df['Calendar Year'] == year]

        self.grouped_df = df_year.groupby(['Region', 'Vehicle Category'])['Population'].sum().reset_index()

        top_regions = self.grouped_df.groupby('Region')['Population'].sum().nlargest(10).index
        top_df = self.grouped_df[self.grouped_df['Region'].isin(top_regions)]

        pivot_df = top_df.pivot(index='Region', columns='Vehicle Category', values='Population').fillna(0)
        pivot_df['Total'] = pivot_df.sum(axis=1)  # Add a 'Total' column
        pivot_df = pivot_df.sort_values(by='Total', ascending=False)  # Sort by 'Total'
        pivot_df = pivot_df.drop(columns='Total')  # Drop the 'Total' column after sorting

        pivot_df.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')

        plt.xlabel("Region", fontsize=14)
        plt.ylabel("Population", fontsize=14)
        plt.title(f"Top 10 Regions by Population in {year}", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Vehicle Category", loc='upper right', fontsize=12, title_fontsize=12)

        plt.tight_layout()
        plt.show()

# %% Q2


class Q2():
    def __init__(self, df):
        self.df = df
        self.baseline_emissions = None

    def selection(self):
        eligible_vehicles = self.df[self.df['Model Year'] >= 2014]
        self.baseline_emissions = eligible_vehicles.groupby('Calendar Year').agg(
            total_nox=('NOx_TOTEX', 'sum'),
            total_pm25=('PM2.5_TOTEX', 'sum')
        ).reset_index()
        return self.baseline_emissions

    def apply_reduction(self, row):
        if 2023 <= row['Calendar Year'] <= 2024:
            row['reduced_nox'] = row['total_nox'] * (1 - 0.07)
            row['reduced_pm25'] = row['total_pm25'] * (1 - 0.05)
        elif 2025 <= row['Calendar Year'] <= 2026:
            row['reduced_nox'] = row['total_nox'] * (1 - 0.47)
            row['reduced_pm25'] = row['total_pm25'] * (1 - 0.35)
        elif row['Calendar Year'] >= 2027:
            row['reduced_nox'] = row['total_nox'] * (1 - 0.63)
            row['reduced_pm25'] = row['total_pm25'] * (1 - 0.45)
        return row

    def plot_emission_reduction(self):

        data = self.baseline_emissions
        years = data['Calendar Year']
        baseline_nox = data['total_nox']
        reduced_nox = data['reduced_nox']
        baseline_pm25 = data['total_pm25']
        reduced_pm25 = data['reduced_pm25']

        nox_reduction_phases = [
            (2023, 2024, '7% Reduction in NOx', 'orange', 0.07),
            (2025, 2026, '47% Reduction in NOx', 'blue', 0.47),
            (2027, 2050, '63% Reduction in NOx', 'green', 0.63)
        ]
        pm25_reduction_phases = [
            (2023, 2024, '5% Reduction in PM2.5', 'orange', 0.05),
            (2025, 2026, '35% Reduction in PM2.5', 'blue', 0.35),
            (2027, 2050, '45% Reduction in PM2.5', 'green', 0.45)
        ]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].bar(years, baseline_nox, color='grey', label='Baseline NOx Emissions')
        for start, end, label, color, reduction in nox_reduction_phases:
            phase_years = years[(years >= start) & (years <= end)]
            phase_reduced_nox = baseline_nox[(years >= start) & (years <= end)] * (1 - reduction)
            axs[0].plot(phase_years, phase_reduced_nox, color=color, label=label, marker='o')

        axs[0].set_title("NOx Emissions Before and After Regulation", fontsize=16)
        axs[0].set_xlabel("Calendar Year", fontsize=14)
        axs[0].set_ylabel("NOx Emissions (tons/day)", fontsize=14)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[0].legend(fontsize=12)

        axs[1].bar(years, baseline_pm25, color='grey', label='Baseline PM2.5 Emissions')
        for start, end, label, color, reduction in pm25_reduction_phases:
            phase_years = years[(years >= start) & (years <= end)]
            phase_reduced_pm25 = baseline_pm25[(years >= start) & (years <= end)] * (1 - reduction)
            axs[1].plot(phase_years, phase_reduced_pm25, color=color, label=label, marker='o')

        axs[1].set_title("PM2.5 Emissions Before and After Regulation", fontsize=16)
        axs[1].set_xlabel("Calendar Year", fontsize=14)
        axs[1].set_ylabel("PM2.5 Emissions (tons/day)", fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=12)
        axs[1].legend(fontsize=12)

        plt.tight_layout()
        plt.show()
# %% Q3


class Q3():
    def __init__(self, df, shapefile_path):
        self.df = df
        self.shapefile_path = shapefile_path
        self.fleet_averages = None

    def calculate_nox_emission_rates(self):
        self.df_hhdt = self.df[(self.df['Vehicle Category'] == 'HHDT') & (self.df['Calendar Year'].between(2020, 2040))]

        self.aggregated = (
            self.df_hhdt.groupby(['Region', 'Calendar Year', "Model Year", 'Fuel']).apply(
                lambda x: pd.Series({
                    'Total NOx_RUNEX': (x['NOx_RUNEX'] * 1e06).sum(),
                    # 'Total NOx_RUNEX': (x['NOx_TOTEX'] * 1e06).sum(),
                    'Total VMT': x['Total VMT'].sum()
                })
            ).reset_index()
        )

        self.aggregated['Emission_rate'] = self.aggregated['Total NOx_RUNEX'] / self.aggregated['Total VMT']

        self.fleet_averages = (
            self.aggregated.groupby('Calendar Year')
            .apply(lambda x: (x['Emission_rate'] * x['Total VMT']).sum() / x['Total VMT'].sum())
            .reset_index(name='Fleet_Average_NOx_Emission_Rate')
        )

        self.aggregated_region = (
            self.df_hhdt.groupby(['Region', 'Calendar Year', "Model Year", 'Fuel']).apply(
                lambda x: pd.Series({
                    'Total NOx_RUNEX': (x['NOx_RUNEX'] * 1e06).sum(),
                    'Total VMT': x['Total VMT'].sum()
                })
            ).reset_index()
        )

        self.aggregated_region['Emission_rate'] = self.aggregated_region['Total NOx_RUNEX'] / self.aggregated_region['Total VMT']

        self.fleet_averages_region = (
            self.aggregated_region.groupby(['Region', 'Calendar Year'])
            .apply(lambda x: (x['Emission_rate'] * x['Total VMT']).sum() / x['Total VMT'].sum())
            .reset_index(name='Fleet_Average_NOx_Emission_Rate')
        )

        return self.fleet_averages, self.aggregated_region, self.fleet_averages_region

    def plot_fleet_average_nox(self):

        plt.figure(figsize=(10, 6))
        plt.bar(self.fleet_averages['Calendar Year'], self.fleet_averages['Fleet_Average_NOx_Emission_Rate'], color='skyblue')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('NOx Emission Rate (g/mile)', fontsize=14)
        plt.title('Fleet Weighted Average NOx Emission Rate of HHDT (2020-2040)', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_nox_heat_map(self, year):
        self.df_aggr = self.fleet_averages_region[self.fleet_averages_region['Calendar Year'] == year]

        self.df_year_agg = self.df_aggr.groupby("Region")["Fleet_Average_NOx_Emission_Rate"].sum()

        self.gdf_counties = gpd.read_file(self.shapefile_path)

        self.gdf_california = self.gdf_counties[self.gdf_counties["STATEFP"] == "06"]

        self.gdf_merged = self.gdf_california.merge(self.df_year_agg, left_on="NAME", right_on="Region", how="left")

        max_nox_value = max(self.df_year_agg.max(), 4)
        norm = plt.Normalize(vmin=0, vmax=max_nox_value)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot = self.gdf_merged.plot(column='Fleet_Average_NOx_Emission_Rate', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=False, norm=norm)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("NOx Emission Rate (g/mile)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        plt.title(f"Weighted Average NOx Emission Rate by County in California ({year})", fontsize=18)

        self.top_10_counties = self.df_year_agg.nlargest(10).index.tolist()
        self.top_10_emissions = self.df_year_agg.nlargest(10).values

        self.top_10_text = "\n".join([f"{county}: {emission:.2f}" for county, emission in zip(self.top_10_counties, self.top_10_emissions)])

        plt.text(
            0.01, 0.02, f"Top 10 Counties by\nNOx Emission Rate (g/mile) in {year}:\n{self.top_10_text}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_yearly_average_nox_heat_map(self, year):

        df_yearly_avg_nox = self.aggregated_region[self.aggregated_region["Calendar Year"] == year].groupby("Region")["Emission_rate"].mean()

        gdf_counties = gpd.read_file(self.shapefile_path)

        gdf_california = gdf_counties[gdf_counties["STATEFP"] == "06"]

        gdf_merged = gdf_california.merge(df_yearly_avg_nox, left_on="NAME", right_on="Region", how="left")

        max_nox_value = max(df_yearly_avg_nox.max(), 8)
        norm = plt.Normalize(vmin=0, vmax=max_nox_value)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot = gdf_merged.plot(column='Emission_rate', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=False, norm=norm)

        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Average NOx Emission Rate (g/mile)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        plt.title(f"Arithmetic Average NOx Emission Rate by County in California ({year})", fontsize=18)

        top_10_counties = df_yearly_avg_nox.nlargest(10).index.tolist()
        top_10_emissions = df_yearly_avg_nox.nlargest(10).values

        top_10_text = "\n".join([f"{county}: {emission:.2f}" for county, emission in zip(top_10_counties, top_10_emissions)])

        plt.text(
            0.00, 0.02, f"Top 10 Counties by\nNOx Emission Rate (g/mile) in {year}:\n{top_10_text}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )

        plt.axis('off')
        plt.tight_layout()
        plt.show()
# %% Q4


class Q4():
    def __init__(self, df):
        self.df = df
        self.add_emission_rates()

    def get_summary_statistics(self):
        return self.df.describe()

    def plot_vmt_inconsistencies(self):
        vmt_inconsistencies = self.df[self.df['Total VMT'] < (self.df['CVMT'] + self.df['EVMT'])].copy()
        vmt_inconsistencies = vmt_inconsistencies[["Region", "Calendar Year", "Vehicle Category", "Model Year", "Total VMT", "CVMT", "EVMT"]]

        vmt_inconsistencies["Difference"] = (vmt_inconsistencies["CVMT"] + vmt_inconsistencies["EVMT"]) - vmt_inconsistencies["Total VMT"]

        pivot_df = vmt_inconsistencies.pivot_table(
            values="Difference", index="Vehicle Category", columns="Model Year", aggfunc="sum"
        )

        categories = pivot_df.index
        model_years = pivot_df.columns
        num_categories = len(categories)
        bar_width = 0.25  # Width of each bar
        margin = 1.0  # Space between vehicle categories
        x_positions = np.arange(num_categories) * (len(model_years) * bar_width + margin)  # Adjust for margin
        colors = cm.viridis(np.linspace(0, 1, len(model_years)))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, model_year in enumerate(model_years):
            ax.barh(
                x_positions + i * bar_width,
                pivot_df[model_year],
                height=bar_width,
                label=f'Model Year {model_year}',
                color=colors[i]
            )

        ax.set_yticks(x_positions + (len(model_years) - 1) * bar_width / 2)
        ax.set_yticklabels(categories)
        ax.set_title("VMT Inconsistencies by Vehicle Category and Model Year", fontsize=16)
        ax.set_xlabel("Difference (CVMT + EVMT - Total VMT)", fontsize=14)
        ax.set_ylabel("Vehicle Category", fontsize=14)
        ax.legend(title="Model Year", loc="lower right", fontsize=12, title_fontsize=14)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        return vmt_inconsistencies

    def analyze_fuel_inconsistencies(self):
        fuel_inconsistencies = self.df[self.df["Fuel"] != "Electricity"]
        fuel_inconsistencies = fuel_inconsistencies[fuel_inconsistencies["Fuel Consumption"] == 0]

        unique_regions = fuel_inconsistencies['Region'].unique()
        inconsistencies_by_region = fuel_inconsistencies['Region'].value_counts()
        vmt_range = (fuel_inconsistencies['Total VMT'].min(), fuel_inconsistencies['Total VMT'].max())
        calendar_year_range = (fuel_inconsistencies['Calendar Year'].min(), fuel_inconsistencies['Calendar Year'].max())
        model_year_range = (fuel_inconsistencies['Model Year'].min(), fuel_inconsistencies['Model Year'].max())
        inconsistencies_by_category = fuel_inconsistencies['Vehicle Category'].value_counts()

        return {
            "Unique Regions": unique_regions,
            "Inconsistencies by Region": inconsistencies_by_region,
            "Total VMT Range": vmt_range,
            "Calendar Year Range": calendar_year_range,
            "Model Year Range": model_year_range,
            "Inconsistencies by Vehicle Category": inconsistencies_by_category
        }

    def plot_energy_inconsistencies_heatmap(self):
        vehicle_categories = self.df["Vehicle Category"].unique()

        category_prompt = "Please select a vehicle category:\n"
        for i, category in enumerate(vehicle_categories, 1):
            category_prompt += f"{i}. {category}\n"

        while True:
            try:
                category_choice = int(input(category_prompt)) - 1
                if 0 <= category_choice < len(vehicle_categories):
                    vehicle_category = vehicle_categories[category_choice]
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        vmt_options = ["EVMT", "Total VMT"]

        vmt_prompt = "Please select the VMT option:\n"
        for i, vmt in enumerate(vmt_options, 1):
            vmt_prompt += f"{i}. {vmt}\n"

        while True:
            try:
                vmt_choice = int(input(vmt_prompt)) - 1
                if 0 <= vmt_choice < len(vmt_options):
                    vmt_column = vmt_options[vmt_choice]
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        energy_inconsistencies = self.df[self.df["Fuel"] == "Electricity"].copy()

        energy_inconsistencies["kWh_mile"] = energy_inconsistencies["Energy Consumption"] / energy_inconsistencies[vmt_column]

        category_data = energy_inconsistencies[energy_inconsistencies["Vehicle Category"] == vehicle_category]

        pivot_table = category_data.pivot_table(
            values="kWh_mile", index="Model Year", columns="Calendar Year", aggfunc="mean"
        )

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            pivot_table,
            cmap="YlGnBu",
            annot=False,
            cbar_kws={"label": "kWh/mile"}
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # Set tick font size
        cbar.set_label("kWh/mile", fontsize=16)  # Set label font size

        plt.title(f"Energy Efficiency (kWh/mile) for {vehicle_category} using {vmt_column}", fontsize=16)
        plt.xlabel("Calendar Year", fontsize=16)
        plt.ylabel("Model Year", fontsize=16)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        plt.show()

        return category_data

    def add_emission_rates(self):
        self.df["emr_pm2.5"] = (self.df["PM2.5_TOTAL"] / self.df["Total VMT"]) * 1e6
        self.df["emr_pm10"] = (self.df["PM10_TOTAL"] / self.df["Total VMT"]) * 1e6
        self.df["emr_nox"] = (self.df["NOx_TOTEX"] / self.df["Total VMT"]) * 1e6
        self.df["emr_c02"] = (self.df["CO2_TOTEX"] / self.df["Total VMT"]) * 1e6
        self.df["emr_ch4"] = (self.df["CH4_TOTEX"] / self.df["Total VMT"]) * 1e6
        self.df["emr_n2o"] = (self.df["N2O_TOTEX"] / self.df["Total VMT"]) * 1e6
        self.df["emr_co"] = (self.df["CO_TOTEX"] / self.df["Total VMT"]) * 1e6
        self.df["emr_SOx"] = (self.df["SOx_TOTEX"] / self.df["Total VMT"]) * 1e6

    def plot_emission_heatmap(self, column_name, vehicle_category, display_mode="weighted_average"):

        df_filtered = self.df[self.df['Vehicle Category'] == vehicle_category].copy()

        if display_mode == "weighted_average":
            df_filtered['weighted_emission'] = df_filtered[column_name] * df_filtered['Total VMT']
            pivot_table = df_filtered.groupby(['Model Year', 'Calendar Year']).apply(
                lambda x: x['weighted_emission'].sum() / x['Total VMT'].sum()
            ).reset_index(name=f'weighted_{column_name}')
            pivot_table = pivot_table.pivot(index='Model Year', columns='Calendar Year', values=f'weighted_{column_name}')
            label = f"Weighted {column_name} (g/mile)"
        elif display_mode == "normal_average":
            pivot_table = df_filtered.groupby(['Model Year', 'Calendar Year'])[column_name].mean().reset_index()
            pivot_table = pivot_table.pivot(index='Model Year', columns='Calendar Year', values=column_name)
            label = f"Normal Average {column_name}"
        elif display_mode == "total":
            pivot_table = df_filtered.groupby(['Model Year', 'Calendar Year'])[column_name].sum().reset_index()
            pivot_table = pivot_table.pivot(index='Model Year', columns='Calendar Year', values=column_name)
            label = f"Total {column_name} (tons/day)"
        else:
            raise ValueError("Invalid display_mode. Choose from 'weighted_average', 'normal_average', or 'total'.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap="YlGnBu", annot=False, cbar_kws={'label': label})

        plt.title(f'Heatmap of {label} for {vehicle_category} by Calendar Year and Model Year')
        plt.xlabel('Calendar Year')
        plt.ylabel('Model Year')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def get_user_selection(self, options, prompt):
        options_prompt = f"{prompt}\n"
        for i, option in enumerate(options, 1):
            options_prompt += f"{i}. {option}\n"

        while True:
            try:
                selection = input(f"{options_prompt}Enter the numbers corresponding to your choices, separated by commas: ")
                selected_indices = [int(x.strip()) - 1 for x in selection.split(",")]
                if all(0 <= idx < len(options) for idx in selected_indices):
                    return [options[idx] for idx in selected_indices]
                else:
                    print("Invalid choice. Please select numbers from the list.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")

    def run_analysis(self):
        vehicle_categories = self.df['Vehicle Category'].unique()
        selected_vehicle_categories = self.get_user_selection(vehicle_categories, "Select the vehicle category you want to draw:")

        emission_types1 = ['emr_pm2.5', 'emr_pm10', 'emr_nox', 'emr_c02', 'emr_ch4', 'emr_n2o', 'emr_co', 'emr_SOx']
        emission_types2 = ["PM2.5_TOTAL", "PM10_TOTAL", "PM2.5_TOTEX", "PM10_TOTEX", 'NOx_TOTEX', 'CO2_TOTEX', 'CH4_TOTEX', 'N2O_TOTEX', 'CO_TOTEX', 'SOx_TOTEX']
        selected_emissions1 = self.get_user_selection(emission_types1, "Select the emission rates you want to draw (normal or weighted average):")
        selected_emissions2 = self.get_user_selection(emission_types2, "Select the emission totals you want to draw:")

        for vehicle_category in selected_vehicle_categories:
            for emission_type in selected_emissions2:
                self.plot_emission_heatmap(emission_type, vehicle_category, "total")

            for emission_type in selected_emissions1:
                self.plot_emission_heatmap(emission_type, vehicle_category, "normal_average")
                self.plot_emission_heatmap(emission_type, vehicle_category, "weighted_average")

    def analyze_gas_data(self):
        self.gas_input_df = self.df[(self.df["Fuel"] != "Electricity") & (self.df["Total VMT"] > 10)]

        column_names = self.gas_input_df.columns[12:].tolist()

        options_prompt = "Available Columns for Analysis:\n"
        for i, column in enumerate(column_names, 1):
            options_prompt += f"{i}. {column}\t"
            if i % 3 == 0:
                options_prompt += "\n"
        options_prompt += "\n" if len(column_names) % 3 != 0 else ""

        while True:
            try:
                column_choice = int(input(f"{options_prompt}Enter the number corresponding to your column choice: ")) - 1
                if 0 <= column_choice < len(column_names):
                    selected_column = column_names[column_choice]
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        zero_count = (self.gas_input_df[selected_column] == 0).sum()
        total_count = len(self.gas_input_df)
        zero_percentage = (zero_count / total_count) * 100

        print(f"\nSelected Column: {selected_column}")
        print(f"Number of rows with zero values in '{selected_column}': {zero_count}")
        print(f"Percentage of rows with zero values: {zero_percentage:.2f}%")

        return {
            "Selected Column": selected_column,
            "Zero Count": zero_count,
            "Zero Percentage": zero_percentage
        }
