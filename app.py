import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="DiPi Assistant",
    page_icon="💡",
    layout="wide"
)

# --- Data Transformation Function ---
def transform_data(df):
    """
    Transforms a dataframe with a two-level header into a tidy, long format.
    """
    df = df.set_index(list(df.columns[:2]))
    df_series = df.stack(level=[0, 1], dropna=True)
    df_long = df_series.reset_index()
    df_long.columns = ['sku', 'description', 'sales_type', 'date_str', 'units_sold']
    df_long['sales_type'] = df_long['sales_type'].str.title()
    df_long['date_str'] = df_long['date_str'].astype(str)
    
    # Use Pandas's automatic date parsing. This is the most robust method.
    df_long['date'] = pd.to_datetime(df_long['date_str'], errors='coerce')
    
    # Drop rows where a date could not be parsed
    df_long.dropna(subset=['date'], inplace=True)

    df_long['date'] = df_long['date'].dt.date
    final_df = df_long[['date', 'sku', 'description', 'sales_type', 'units_sold']].sort_values(by='date')
    return final_df

# --- App Title and Description ---
st.title("DiPi - The Demand Planning Assistant")
st.write("Upload your sales data (in wide format) to get insights, visualizations, and a sales forecast.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_data = pd.read_csv(uploaded_file, header=[0, 1])
        else:
            raw_data = pd.read_excel(uploaded_file, header=[0, 1])
        
        data = transform_data(raw_data)
        
        # ⭐️ FIX: Convert description to string to allow numbers and special characters
        data['sku_desc'] = data['sku'].astype(str) + ' - ' + data['description'].astype(str)
        
        data['month_year'] = pd.to_datetime(data['date']).dt.to_period('M').astype(str)
        st.success("Data loaded and transformed successfully!")
        
        st.write("Preview of Transformed Data (Long Format):")
        st.dataframe(data.head())

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.warning("Please ensure your Excel file has a two-row header structure and valid date columns.")
        st.stop()

    sku_desc_pairs = data[['sku', 'description']].drop_duplicates()

    # --- Main Panel with Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["💡 Insights & Analytics", "📈 Visualizations", "🔮 Forecasting", "📥 Export Data"])

    with tab1:
        st.subheader("Key Performance Indicators")

        for index, row in sku_desc_pairs.iterrows():
            sku, description = row['sku'], row['description']
            st.markdown(f"#### {sku} - {description}")
            sku_data = data[data['sku'] == sku]
            total_units = int(sku_data['units_sold'].sum())
            promo_units = int(sku_data[sku_data['sales_type'] == 'Promo Sales']['units_sold'].sum())
            regular_units = total_units - promo_units
            promo_percent = (promo_units / total_units) * 100 if total_units > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Units Sold", f"{total_units:,}")
            col2.metric("Regular Units Sold", f"{regular_units:,}")
            col3.metric("Promotion Units", f"{promo_units:,}")
            col4.metric("Promotion %", f"{promo_percent:.2f}%")
        
        st.markdown("---")
        sku_contribution = data.groupby('sku_desc')['units_sold'].sum().reset_index()
        fig_pie = px.pie(sku_contribution, names='sku_desc', values='units_sold', title='SKU Contribution to Total Sales', hole=.3)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Overall Total Metrics (All SKUs)")
        
        # ⭐️ CHANGE: Removed the st.container(border=True) wrapper from this section
        total_units_sold = int(data['units_sold'].sum())
        total_promo_sales = int(data[data['sales_type'] == 'Promo Sales']['units_sold'].sum())
        total_regular_sales = total_units_sold - total_promo_sales
        promo_percentage = (total_promo_sales / total_units_sold) * 100 if total_units_sold > 0 else 0
        
        col1_total, col2_total, col3_total, col4_total = st.columns(4)
        col1_total.metric("Total Units Sold", f"{total_units_sold:,}")
        col2_total.metric("Regular Units Sold", f"{total_regular_sales:,}")
        col3_total.metric("Total Promotion Units", f"{total_promo_sales:,}")
        col4_total.metric("Promotion % of Sales", f"{promo_percentage:.2f}%")
        
        st.markdown("---")
        st.subheader("Monthly Performance Highlights")
        
        monthly_sales = data.groupby(['sku', 'description', 'month_year'])['units_sold'].sum().reset_index()
        for index, row in sku_desc_pairs.iterrows():
            sku, description = row['sku'], row['description']
            st.markdown(f"**{sku} - {description}**")
            
            sku_monthly_sales = monthly_sales[monthly_sales['sku'] == sku]
            if len(sku_monthly_sales) > 1:
                best_month_record = sku_monthly_sales.loc[sku_monthly_sales['units_sold'].idxmax()]
                worst_month_record = sku_monthly_sales.loc[sku_monthly_sales['units_sold'].idxmin()]
                st.write(f"📈 Best Month: **{best_month_record['month_year']}** ({best_month_record['units_sold']:,} units)")
                st.write(f"📉 Worst Month: **{worst_month_record['month_year']}** ({worst_month_record['units_sold']:,} units)")
            elif not sku_monthly_sales.empty:
                 single_month = sku_monthly_sales.iloc[0]
                 st.write(f"🗓️ Only one month of data: **{single_month['month_year']}** ({single_month['units_sold']:,} units)")
            else:
                 st.write("No sales data found for this SKU.")
        
        st.markdown("---")
        st.subheader("Promotion Uplift Analysis")
        uplift_results = []
        for index, row in sku_desc_pairs.iterrows():
            sku, description = row['sku'], row['description']
            sku_data = data[data['sku'] == sku]
            regular_sales = sku_data[sku_data['sales_type'] == 'Regular Sales']
            
            if not regular_sales.empty and regular_sales['units_sold'].mean() > 0:
                baseline = regular_sales['units_sold'].mean()
                promo_sales = sku_data[sku_data['sales_type'] == 'Promo Sales']
                
                for _, promo_row in promo_sales.iterrows():
                    uplift_percent = ((promo_row['units_sold'] - baseline) / baseline) * 100
                    uplift_results.append({
                        'SKU': sku,
                        'Description': description,
                        'Promo Month': promo_row['month_year'],
                        'Baseline Sales': round(baseline),
                        'Actual Promo Sales': promo_row['units_sold'],
                        'Uplift %': round(uplift_percent, 2)
                    })
        
        if uplift_results:
            uplift_df = pd.DataFrame(uplift_results)
            st.dataframe(uplift_df)
            
            fig_uplift = px.bar(uplift_df, x='Promo Month', y='Uplift %', color='Description', title='Promotion Uplift %')
            fig_uplift.add_hline(y=0, line_dash="solid", line_width=3, line_color="white")
            st.plotly_chart(fig_uplift, use_container_width=True)
        else:
            st.info("No promotional data with a valid baseline found to calculate uplift.")

    with tab2:
        st.subheader("Sales Visualizations")

        sales_type_filter = st.selectbox(
            "Select Sales Type to Display:",
            options=['Total Sales', 'Regular Sales', 'Promo Sales'],
            key='sales_type_filter'
        )

        if sales_type_filter == 'Total Sales':
            chart_data = data.groupby(['date', 'sku_desc'])['units_sold'].sum().reset_index()
        elif sales_type_filter == 'Regular Sales':
            chart_data = data[data['sales_type'] == 'Regular Sales']
        else: # Promo Sales
            chart_data = data[data['sales_type'] == 'Promo Sales']

        fig_time = px.line(chart_data, x='date', y='units_sold', color='sku_desc', title=f'{sales_type_filter} Over Time')
        fig_time.update_yaxes(nticks=5)
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.markdown("---")
        sales_by_type = data.groupby(['sku_desc', 'sales_type'])['units_sold'].sum().reset_index()
        fig_promo = px.bar(sales_by_type, x='sku_desc', y='units_sold', color='sales_type',
                           title='Promotion vs. Regular Sales by SKU', barmode='stack')
        st.plotly_chart(fig_promo, use_container_width=True)

        st.markdown("---")
        st.subheader("Monthly Sales Heatmap")
        
        num_years = pd.to_datetime(data['date']).dt.year.nunique()

        if num_years >= 2:
            heatmap_data = data.copy()
            heatmap_data['date'] = pd.to_datetime(heatmap_data['date'])
            heatmap_data['year'] = heatmap_data['date'].dt.year
            heatmap_data['month'] = heatmap_data['date'].dt.strftime('%b')
            sales_matrix = heatmap_data.pivot_table(index='year', columns='month', values='units_sold', aggfunc='sum')
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            sales_matrix = sales_matrix.reindex(columns=month_order)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=sales_matrix.values,
                x=sales_matrix.columns,
                y=sales_matrix.index,
                colorscale='Viridis'
            ))
            fig_heatmap.update_layout(title='Monthly Sales Volume by Year', xaxis_title='Month', yaxis_title='Year')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("The dataset must contain at least 2 different years of data to generate a heatmap.")

    with tab3:
        with st.expander("ℹ️ What is Prophet?"):
            st.write("""
            Prophet is a powerful and user-friendly forecasting tool developed by Meta (Facebook). 
            It is specifically designed for business time-series data, which often has strong seasonal effects (e.g., weekly, yearly) and holidays. 
            It's robust to missing data and shifts in trends, often producing high-quality forecasts without requiring expert knowledge.
            """)
        
        st.subheader("Configure Your Forecast")
        col1, col2 = st.columns(2)
        with col1:
            selected_sku_display = st.selectbox(
                "1. Select a Product (SKU)",
                options=[f"{row['sku']} - {row['description']}" for index, row in sku_desc_pairs.iterrows()]
            )
            selected_sku = int(selected_sku_display.split(' - ')[0])
            st.session_state['selected_sku'] = selected_sku
        with col2:
            forecast_months = st.number_input(
                "2. Select Forecast Period (months)",
                min_value=1, max_value=12, value=3
            )
        
        forecast_type = st.radio(
            "3. Select Data to Forecast",
            options=['Total Sales', 'Regular Sales', 'Promo Sales'],
            horizontal=True
        )

        st.markdown("---")
        st.subheader(f"{forecast_type} Forecast for: {selected_sku_display}")
        
        if 'forecast_df' not in st.session_state: st.session_state.forecast_df = None
        if 'prophet_model' not in st.session_state: st.session_state.prophet_model = None

        if st.button("Generate Forecast"):
            forecast_periods_in_days = forecast_months * 30
            sku_data = data[data['sku'] == selected_sku].copy()
            
            if forecast_type == 'Total Sales':
                daily_sku_data = sku_data.groupby('date')['units_sold'].sum().reset_index()
            elif forecast_type == 'Regular Sales':
                regular_data = sku_data[sku_data['sales_type'] == 'Regular Sales']
                daily_sku_data = regular_data.groupby('date')['units_sold'].sum().reset_index()
            else: # Promo Sales
                promo_data = sku_data[sku_data['sales_type'] == 'Promo Sales']
                daily_sku_data = promo_data.groupby('date')['units_sold'].sum().reset_index()
            
            daily_sku_data['date'] = pd.to_datetime(daily_sku_data['date'])
            
            if len(daily_sku_data) < 3:
                st.warning(f"Cannot generate a forecast. The selected SKU has fewer than 3 data points for '{forecast_type}'.")
                st.session_state.forecast_df = None
            else:
                with st.spinner("Generating Prophet forecast..."):
                    prophet_df_input = daily_sku_data[['date', 'units_sold']].rename(columns={'date': 'ds', 'units_sold': 'y'})
                    model = Prophet()
                    model.fit(prophet_df_input)
                    future = model.make_future_dataframe(periods=forecast_periods_in_days)
                    forecast = model.predict(future)
                    st.session_state.prophet_model = model
                    st.session_state.forecast_df = forecast
                st.success("Forecast generated!")

        if st.session_state.forecast_df is not None:
            model = st.session_state.prophet_model
            forecast_to_show = st.session_state.forecast_df
            
            granularity = st.radio(
                "Select forecast table view:",
                ['Every 15 Days', 'Monthly'],
                horizontal=True,
                key='granularity_selector'
            )
            future_forecast = forecast_to_show[forecast_to_show['ds'] > pd.to_datetime(data['date'].max())]
            
            if granularity == 'Every 15 Days':
                sampled_forecast = future_forecast.iloc[::15]
            else: # Monthly
                sampled_forecast = future_forecast.set_index('ds').resample('MS').first().reset_index()

            display_df = sampled_forecast.copy()
            display_df['ds'] = pd.to_datetime(display_df['ds']).dt.date
            numeric_cols = ['yhat', 'yhat_lower', 'yhat_upper']
            display_df[numeric_cols] = display_df[numeric_cols].round(0).astype(int)
            display_df = display_df.rename(columns={
                'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Low Estimate', 'yhat_upper': 'High Estimate'
            })
            
            st.write(f"Forecast Data ({granularity})")
            st.dataframe(display_df[['Date', 'Forecast', 'Low Estimate', 'High Estimate']])
            
            st.write("Forecast Visualization")
            fig_forecast = plot_plotly(model, forecast_to_show)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.write("Forecast Components")
            fig_components = plot_components_plotly(model, forecast_to_show)
            
            trend_min = forecast_to_show['trend'].min()
            trend_max = forecast_to_show['trend'].max()
            padding = (trend_max - trend_min) * 0.20
            fig_components.update_layout(yaxis_range=[trend_min - padding, trend_max + padding])
            
            st.plotly_chart(fig_components, use_container_width=True)
        else:
            st.info("Click the button above to generate a forecast.")
    
    with tab4:
        st.subheader("Export Your Data")
        st.download_button(
           label="Download Transformed Data as CSV",
           data=data.to_csv(index=False).encode('utf-8'),
           file_name='transformed_sales_data.csv',
           mime='text/csv',
        )
        st.markdown("---")
        st.write("Download the full forecast results for the selected SKU.")
        if st.session_state.forecast_df is not None:
            export_sku = st.session_state.get('selected_sku', 'forecast')
            st.download_button(
               label="Download Forecast Data as CSV",
               data=st.session_state.forecast_df.to_csv(index=False).encode('utf-8'),
               file_name=f'forecast_{export_sku}.csv',
               mime='text/csv',
            )
        else:
            st.warning("You must generate a forecast in the 'Forecasting' tab before you can download it.")
