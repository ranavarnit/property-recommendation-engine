# real_estate_recommendation.py
# Property Recommendation Engine - Enhanced Version
# Recommends properties based on user preferences

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

class PropertyRecommender:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.properties_df = None
        self.similarity_matrix = None

    def load_data(self, filepath):
        """Load property dataset"""
        df = pd.read_csv(filepath)
        return df

    def extract_bhk(self, size_str):
        """Extract numeric BHK/bedroom count from size string like '2 BHK', '4 Bedroom', '1 RK'"""
        if pd.isna(size_str):
            return np.nan
        size_str = str(size_str).strip()
        match = re.search(r'(\d+)', size_str)
        if match:
            return int(match.group(1))
        return np.nan

    def preprocess(self, df):
        """Preprocess data for recommendation"""
        df = df.copy()

        # Handle total_sqft - convert range strings to average, handle non-numeric values
        if 'total_sqft' in df.columns:
            def convert_sqft_to_num(x):
                if pd.isna(x):
                    return np.nan
                x_str = str(x).strip()
                tokens = x_str.split('-')
                if len(tokens) == 2:
                    try:
                        return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
                    except:
                        pass
                try:
                    return float(x_str)
                except:
                    num_match = re.search(r'[\d.]+', x_str)
                    if num_match:
                        try:
                            return float(num_match.group(0))
                        except:
                            return np.nan
                    return np.nan

            df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
            df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())

        # Extract numeric BHK from size column
        if 'size' in df.columns:
            df['size_num'] = df['size'].apply(self.extract_bhk)
            df['size_num'] = df['size_num'].fillna(df['size_num'].median())

        # Fill missing values for numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Create feature matrix for similarity
        feature_cols = ['size_num', 'total_sqft', 'bath', 'balcony', 'price'] + [col + '_enc' for col in categorical_cols]
        available_feature_cols = [col for col in feature_cols if col in df.columns]

        self.feature_matrix = df[available_feature_cols].values

        if np.isnan(self.feature_matrix).any():
            col_means = np.nanmean(self.feature_matrix, axis=0)
            inds = np.where(np.isnan(self.feature_matrix))
            self.feature_matrix[inds] = np.take(col_means, inds[1])

        self.scaler.fit(self.feature_matrix)
        self.scaled_features = self.scaler.transform(self.feature_matrix)
        self.similarity_matrix = cosine_similarity(self.scaled_features)
        self.properties_df = df
        return df

    def recommend(self, budget, location=None, bhk=None, top_n=5):
        """Recommend properties based on user preferences"""
        df = self.properties_df.copy()

        # Filter by budget (±20%)
        min_budget = budget * 0.8
        max_budget = budget * 1.2
        df = df[(df['price'] >= min_budget) & (df['price'] <= max_budget)]

        # Filter by location if specified
        if location and location != 'Any':
            df = df[df['location'] == location]

        # Filter by BHK if specified
        if bhk and bhk != 'Any':
            try:
                bhk_num = int(bhk)
                df = df[df['size_num'] == bhk_num]
            except (ValueError, TypeError):
                bhk_num = self.extract_bhk(bhk)
                if not pd.isna(bhk_num):
                    df = df[df['size_num'] == bhk_num]

        if len(df) == 0:
            return pd.DataFrame()

        # Sort by price (closest to budget first)
        df['price_diff'] = abs(df['price'] - budget)
        df = df.sort_values('price_diff')
        return df.head(top_n)

    def get_price_trends(self, location=None):
        """Get price trends by location"""
        df = self.properties_df.copy()

        if location and location != 'Any':
            df = df[df['location'] == location]

        df['price_per_sqft'] = df['price'] / df['total_sqft']

        trends = df.groupby('location').agg({
            'price': ['mean', 'median', 'min', 'max'],
            'price_per_sqft': 'mean',
            'total_sqft': 'mean'
        }).round(2)
        return trends

    def get_unique_bhk_options(self):
        """Get sorted unique BHK options for dropdown"""
        if self.properties_df is None or 'size_num' not in self.properties_df.columns:
            return ['Any']
        bhk_values = sorted(self.properties_df['size_num'].dropna().unique().astype(int).tolist())
        return ['Any'] + [str(x) for x in bhk_values]

    def get_price_stats(self):
        """Get overall price statistics"""
        df = self.properties_df
        return {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
            'median': df['price'].median()
        }


def format_price(price):
    """Format price for display"""
    if price >= 100:
        return f"₹{price/100:.2f} Cr"
    else:
        return f"₹{price:.2f} L"


def run_dashboard():
    st.set_page_config(page_title="Property Recommendation Engine", layout="wide")

    st.title("🏠 Property Recommendation Engine")
    st.markdown("Find your perfect property based on budget and preferences!")

    @st.cache_resource
    def init_recommender():
        recommender = PropertyRecommender()
        df = recommender.load_data("Bengaluru_House_Data.csv")
        recommender.preprocess(df)
        return recommender

    try:
        recommender = init_recommender()
        price_stats = recommender.get_price_stats()

        # Sidebar controls
        st.sidebar.header("Your Preferences")

        # Show price range info
        st.sidebar.markdown(f"**Dataset Price Range:**")
        st.sidebar.markdown(f"Min: {format_price(price_stats['min'])} | Max: {format_price(price_stats['max'])}")
        st.sidebar.markdown(f"Median: {format_price(price_stats['median'])}")
        st.sidebar.markdown("---")

        # Budget slider with better range based on actual data
        min_price = max(10, int(price_stats['min'] * 0.5))
        max_price = min(5000, int(price_stats['max'] * 1.2))
        default_budget = int(price_stats['median'])

        # Handle reset: set defaults in session state before creating widgets
        if 'reset_triggered' in st.session_state and st.session_state['reset_triggered']:
            st.session_state['budget_slider'] = default_budget
            st.session_state['location_select'] = 'Any'
            st.session_state['bhk_select'] = 'Any'
            st.session_state['top_n_slider'] = 5
            st.session_state['reset_triggered'] = False

        # Use explicit keys for all widgets
        budget_lakhs = st.sidebar.slider(
            "Budget (₹ Lakhs)", 
            min_value=min_price, 
            max_value=max_price, 
            value=default_budget, 
            step=10,
            key='budget_slider'
        )
        budget = budget_lakhs

        locations = ['Any'] + sorted(recommender.properties_df['location'].dropna().unique().tolist())
        location = st.sidebar.selectbox(
            "Preferred Location", 
            locations,
            index=0,  # 'Any' is first
            key='location_select'
        )

        size_options = recommender.get_unique_bhk_options()
        bhk = st.sidebar.selectbox(
            "BHK", 
            size_options,
            index=0,  # 'Any' is first
            key='bhk_select'
        )

        top_n = st.sidebar.slider(
            "Number of Recommendations", 
            1, 10, 
            5,
            key='top_n_slider'
        )

        # Add a reset button - sets flag to reset values on next rerun
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Reset Filters", key='reset_button'):
            st.session_state['reset_triggered'] = True
            st.rerun()

        # Get recommendations
        st.header("Recommended Properties")
        recommendations = recommender.recommend(budget, location, bhk, top_n)

        if len(recommendations) > 0:
            st.success(f"Found **{len(recommendations)}** properties matching your criteria!")

            # Create formatted display dataframe
            display_df = recommendations.copy()

            # Format columns for display
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(format_price)
            if 'total_sqft' in display_df.columns:
                display_df['total_sqft'] = display_df['total_sqft'].apply(lambda x: f"{x:,.0f} sqft")
            if 'bath' in display_df.columns:
                display_df['bath'] = display_df['bath'].apply(lambda x: f"{int(x)}")
            if 'balcony' in display_df.columns:
                display_df['balcony'] = display_df['balcony'].apply(lambda x: f"{int(x)}")
            if 'size' in display_df.columns:
                display_df['size'] = display_df['size'].fillna('N/A')
            if 'area_type' in display_df.columns:
                display_df['area_type'] = display_df['area_type'].fillna('N/A')

            # Reorder columns for better display
            preferred_cols = ['location', 'size', 'total_sqft', 'bath', 'balcony', 'price', 'area_type']
            available_cols = [col for col in preferred_cols if col in display_df.columns]
            display_df = display_df[available_cols]

            # Style the dataframe
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Visualizations
            st.markdown("---")

            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Properties Found", len(recommendations))
            with col2:
                if 'price' in recommendations.columns:
                    avg_price = recommendations['price'].mean()
                    st.metric("Avg Price", format_price(avg_price))
            with col3:
                if 'total_sqft' in recommendations.columns:
                    avg_sqft = recommendations['total_sqft'].mean()
                    st.metric("Avg Size", f"{avg_sqft:,.0f} sqft")
            with col4:
                if 'bath' in recommendations.columns:
                    avg_bath = recommendations['bath'].mean()
                    st.metric("Avg Baths", f"{avg_bath:.1f}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Price Distribution")
                if len(recommendations) > 0:
                    # For small or tightly clustered datasets, use a box plot or violin plot instead
                    if len(recommendations) <= 3 or (recommendations['price'].max() - recommendations['price'].min()) < 5:
                        fig = px.box(
                            recommendations, 
                            y='price',
                            title='Price Range of Recommendations',
                            labels={'price': 'Price (₹ Lakhs)'},
                            color_discrete_sequence=['#636EFA'],
                            points='all'
                        )
                        fig.update_traces(marker_size=10)
                    else:
                        fig = px.histogram(
                            recommendations, 
                            x='price', 
                            nbins=min(20, len(recommendations)),
                            title='Price Distribution of Recommendations',
                            labels={'price': 'Price (₹ Lakhs)', 'count': 'Number of Properties'},
                            color_discrete_sequence=['#636EFA']
                        )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No properties to display.")

            with col2:
                st.subheader("Properties by Location")
                if 'location' in recommendations.columns and len(recommendations) > 0:
                    loc_counts = recommendations['location'].value_counts()
                    if len(loc_counts) > 0:
                        fig = px.pie(
                            values=loc_counts.values, 
                            names=loc_counts.index, 
                            title='Properties by Location',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Price per sqft analysis
            st.markdown("---")
            st.subheader("Price Analysis")

            if 'total_sqft' in recommendations.columns and 'price' in recommendations.columns:
                recommendations['price_per_sqft'] = recommendations['price'] / recommendations['total_sqft']

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(
                        recommendations,
                        x='total_sqft',
                        y='price',
                        color='location' if 'location' in recommendations.columns else None,
                        size='size_num' if 'size_num' in recommendations.columns else None,
                        hover_data=['size', 'area_type'] if 'size' in recommendations.columns and 'area_type' in recommendations.columns else None,
                        title='Price vs Total Sqft',
                        labels={'total_sqft': 'Total Sqft', 'price': 'Price (₹ Lakhs)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        recommendations.sort_values('price_per_sqft', ascending=False),
                        x='location' if 'location' in recommendations.columns else recommendations.index,
                        y='price_per_sqft',
                        title='Price per Sqft by Location',
                        labels={'price_per_sqft': 'Price per Sqft (₹ Lakhs)', 'location': 'Location'},
                        color='price_per_sqft',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No properties found matching your criteria. Try adjusting your budget or preferences.")

            # Show suggestions
            st.markdown("**Suggestions:**")
            st.markdown("- Try increasing your budget range")
            st.markdown("- Select 'Any' for location or BHK")
            st.markdown("- Check the dataset price range in the sidebar")

        # Price trends
        st.markdown("---")
        st.header("Price Trends by Location")

        with st.spinner("Loading price trends..."):
            trends = recommender.get_price_trends(location if location != 'Any' else None)
            if len(trends) > 0:
                # Flatten multi-level columns
                trends.columns = [' '.join(col).strip() for col in trends.columns.values]
                trends = trends.reset_index()
                st.dataframe(trends, use_container_width=True, hide_index=True)

                # Show top locations by average price
                if 'price mean' in trends.columns and 'location' in trends.columns:
                    st.subheader("Top Locations by Average Price")
                    top_locations = trends.nlargest(10, 'price mean')

                    if len(top_locations) <= 1:
                        st.info("Only one location matches your current filter. Showing all locations instead.")
                        # Show all locations for comparison context
                        all_locations = recommender.get_price_trends().nlargest(10, 'price mean')
                        all_locations.columns = [' '.join(col).strip() for col in all_locations.columns.values]
                        all_locations = all_locations.reset_index()
                        if 'price mean' in all_locations.columns:
                            fig = px.bar(
                                all_locations,
                                x='location',
                                y='price mean',
                                title='Top 10 Locations by Average Price (All Locations)',
                                labels={'price mean': 'Average Price (₹ Lakhs)', 'location': 'Location'},
                                color='price mean',
                                color_continuous_scale='Plasma'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.bar(
                            top_locations,
                            x='location',
                            y='price mean',
                            title='Top 10 Locations by Average Price',
                            labels={'price mean': 'Average Price (₹ Lakhs)', 'location': 'Location'},
                            color='price mean',
                            color_continuous_scale='Plasma'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trend data available for the selected filters.")

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Bengaluru_House_Data.csv' is in the same directory.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    run_dashboard()
