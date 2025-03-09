import pandas as pd
import os
import logging
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """DataLoader class for loading and managing bond data"""

    def __init__(self):
        """Initialize DataLoader with empty dataframes"""
        self.bond_details = None
        self.cashflow_details = None
        self.company_insights = None
        self.loaded_files = []

    def load_bond_details(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load bond details from CSV file

        Args:
            file_path: Path to the bond details CSV file

        Returns:
            DataFrame with bond details or None if loading fails
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None

            df = pd.read_csv(file_path)

            if df.empty:
                logger.error(f"Empty dataframe loaded from {file_path}")
                return None

            # If this is the first bond file
            if self.bond_details is None:
                self.bond_details = df
            else:
                # Concatenate with existing bond details
                self.bond_details = pd.concat([self.bond_details, df], ignore_index=True)
                # Remove duplicates based on ISIN
                self.bond_details = self.bond_details.drop_duplicates(subset=['isin'], keep='first')

            self.loaded_files.append(file_path)
            logger.info(f"Successfully loaded bond details from {file_path}, total bonds: {len(self.bond_details)}")
            return self.bond_details

        except Exception as e:
            logger.error(f"Error loading bond details from {file_path}: {str(e)}")
            return None

    def load_cashflow_details(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load cashflow details from CSV file

        Args:
            file_path: Path to the cashflow details CSV file

        Returns:
            DataFrame with cashflow details or None if loading fails
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None

            df = pd.read_csv(file_path)

            if df.empty:
                logger.error(f"Empty dataframe loaded from {file_path}")
                return None

            self.cashflow_details = df
            self.loaded_files.append(file_path)
            logger.info(f"Successfully loaded cashflow details from {file_path}, total records: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Error loading cashflow details from {file_path}: {str(e)}")
            return None

    def load_company_insights(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load company insights from CSV file

        Args:
            file_path: Path to the company insights CSV file

        Returns:
            DataFrame with company insights or None if loading fails
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None

            df = pd.read_csv(file_path)

            if df.empty:
                logger.error(f"Empty dataframe loaded from {file_path}")
                return None

            self.company_insights = df
            self.loaded_files.append(file_path)
            logger.info(f"Successfully loaded company insights from {file_path}, total companies: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Error loading company insights from {file_path}: {str(e)}")
            return None

    def get_bond_by_isin(self, isin: str) -> Optional[Dict[str, Any]]:
        """
        Get bond details by ISIN

        Args:
            isin: The ISIN code

        Returns:
            Dictionary with bond details or None if not found
        """
        try:
            if self.bond_details is None:
                logger.warning("Bond details not loaded")
                return None

            matching_bonds = self.bond_details[self.bond_details['isin'] == isin]

            if matching_bonds.empty:
                logger.warning(f"No bond found with ISIN: {isin}")
                return None

            bond_data = matching_bonds.iloc[0].to_dict()

            # Parse any JSON fields
            for field in ['coupon_details', 'issuer_details']:
                if field in bond_data and isinstance(bond_data[field], str):
                    try:
                        bond_data[field] = json.loads(bond_data[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON for field {field}")

            return bond_data

        except Exception as e:
            logger.error(f"Error getting bond by ISIN {isin}: {str(e)}")
            return None

    def get_bonds_by_company(self, company_name: str) -> List[Dict[str, Any]]:
        """
        Get bonds by company name

        Args:
            company_name: The company name

        Returns:
            List of bond dictionaries
        """
        try:
            if self.bond_details is None:
                logger.warning("Bond details not loaded")
                return []

            matching_bonds = self.bond_details[
                self.bond_details['company_name'].str.contains(company_name, case=False, na=False)
            ]

            if matching_bonds.empty:
                logger.warning(f"No bonds found for company: {company_name}")
                return []

            bonds_list = []
            for _, row in matching_bonds.iterrows():
                bond_data = row.to_dict()
                # Parse any JSON fields
                for field in ['coupon_details', 'issuer_details']:
                    if field in bond_data and isinstance(bond_data[field], str):
                        try:
                            bond_data[field] = json.loads(bond_data[field])
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON for field {field}")

                bonds_list.append(bond_data)

            return bonds_list

        except Exception as e:
            logger.error(f"Error getting bonds by company {company_name}: {str(e)}")
            return []

    def get_cashflows_by_isin(self, isin: str) -> List[Dict[str, Any]]:
        """
        Get cashflows by ISIN

        Args:
            isin: The ISIN code

        Returns:
            List of cashflow dictionaries
        """
        try:
            if self.cashflow_details is None:
                logger.warning("Cashflow details not loaded")
                return []

            matching_cashflows = self.cashflow_details[self.cashflow_details['isin'] == isin]

            if matching_cashflows.empty:
                logger.warning(f"No cashflows found for ISIN: {isin}")
                return []

            return matching_cashflows.to_dict('records')

        except Exception as e:
            logger.error(f"Error getting cashflows by ISIN {isin}: {str(e)}")
            return []

    def get_company_by_name(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Get company insights by name

        Args:
            company_name: The company name

        Returns:
            Dictionary with company insights or None if not found
        """
        try:
            if self.company_insights is None:
                logger.warning("Company insights not loaded")
                return None

            matching_companies = self.company_insights[
                self.company_insights['company_name'].str.contains(company_name, case=False, na=False)
            ]

            if matching_companies.empty:
                logger.warning(f"No company found with name: {company_name}")
                return None

            company_data = matching_companies.iloc[0].to_dict()

            # Parse any JSON fields
            for field in ['key_metrics', 'balance_sheet', 'income_statement', 'cashflow']:
                if field in company_data and isinstance(company_data[field], str):
                    try:
                        company_data[field] = json.loads(company_data[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON for field {field}")

            return company_data

        except Exception as e:
            logger.error(f"Error getting company by name {company_name}: {str(e)}")
            return None

    def search_bonds(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search bonds by term

        Args:
            search_term: The search term

        Returns:
            List of matching bond dictionaries
        """
        try:
            if self.bond_details is None:
                logger.warning("Bond details not loaded")
                return []

            # Search in ISIN and company_name
            matching_bonds = self.bond_details[
                (self.bond_details['isin'].str.contains(search_term, case=False, na=False)) |
                (self.bond_details['company_name'].str.contains(search_term, case=False, na=False))
            ]

            if matching_bonds.empty:
                logger.warning(f"No bonds found matching: {search_term}")
                return []

            bonds_list = []
            for _, row in matching_bonds.iterrows():
                bond_data = row.to_dict()
                # Parse any JSON fields
                for field in ['coupon_details', 'issuer_details']:
                    if field in bond_data and isinstance(bond_data[field], str):
                        try:
                            bond_data[field] = json.loads(bond_data[field])
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON for field {field}")

                bonds_list.append(bond_data)

            return bonds_list

        except Exception as e:
            logger.error(f"Error searching bonds with term {search_term}: {str(e)}")
            return []

    def load_multiple_bond_details(self, filepaths: List[str]) -> pd.DataFrame:
        """
        Load multiple bond details CSV files at once.

        Args:
            filepaths: List of paths to bond_details.csv files

        Returns:
            DataFrame containing combined bond details
        """
        for filepath in filepaths:
            try:
                self.load_bond_details(filepath)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error loading {filepath}: {str(e)}")

        return self.bond_details

    def load_all_data(self, bond_filepath: str, company_filepath: str, cashflow_filepath: str) -> None:
        """
        Load all datasets at once.

        Args:
            bond_filepath: Path to bond_details.csv
            company_filepath: Path to company_insights.csv
            cashflow_filepath: Path to cashflow_details.csv
        """
        self.load_bond_details(bond_filepath)
        self.load_company_insights(company_filepath)
        self.load_cashflow_details(cashflow_filepath)

    def filter_bonds_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict]:
        """
        Filter bonds based on multiple criteria.

        Args:
            criteria: Dictionary with filter criteria

        Returns:
            List of dictionaries with filtered bond details
        """
        if self.bond_details is None:
            return []

        filtered_df = self.bond_details.copy()

        # Fill NaN values in key columns to prevent filtering errors
        for column in filtered_df.columns:
            if column in ['isin', 'company_name', 'maturity_date']:
                filtered_df[column] = filtered_df[column].fillna('')

        # Apply filters
        for key, value in criteria.items():
            if key in filtered_df.columns:
                # Handle NaN values before filtering
                if pd.api.types.is_numeric_dtype(filtered_df[key]):
                    filtered_df[key] = filtered_df[key].fillna(0)
                elif filtered_df[key].dtype == 'object':
                    filtered_df[key] = filtered_df[key].fillna('')

                if isinstance(value, dict) and 'min' in value and 'max' in value:
                    # Range filter
                    filtered_df = filtered_df[
                        (filtered_df[key] >= value['min']) &
                        (filtered_df[key] <= value['max'])
                    ]
                elif isinstance(value, dict) and 'min' in value:
                    # Min filter
                    filtered_df = filtered_df[filtered_df[key] >= value['min']]
                elif isinstance(value, dict) and 'max' in value:
                    # Max filter
                    filtered_df = filtered_df[filtered_df[key] <= value['max']]
                elif isinstance(value, list):
                    # List of values
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    # Exact match
                    filtered_df = filtered_df[filtered_df[key] == value]
            elif key == 'maturity_after':
                # Fill NaN values before converting to datetime
                filtered_df['maturity_date'] = filtered_df['maturity_date'].fillna('')

                # Filter by maturity date after a specific date
                filtered_df['maturity_date'] = pd.to_datetime(
                    filtered_df['maturity_date'],
                    format='%d-%m-%Y',
                    errors='coerce'
                )
                filter_date = pd.to_datetime(value, format='%d-%m-%Y', errors='coerce')
                # Only include rows with valid dates
                filtered_df = filtered_df[~filtered_df['maturity_date'].isna() & (filtered_df['maturity_date'] > filter_date)]
            elif key == 'maturity_before':
                # Fill NaN values before converting to datetime
                filtered_df['maturity_date'] = filtered_df['maturity_date'].fillna('')

                # Filter by maturity date before a specific date
                filtered_df['maturity_date'] = pd.to_datetime(
                    filtered_df['maturity_date'],
                    format='%d-%m-%Y',
                    errors='coerce'
                )
                filter_date = pd.to_datetime(value, format='%d-%m-%Y', errors='coerce')
                # Only include rows with valid dates
                filtered_df = filtered_df[~filtered_df['maturity_date'].isna() & (filtered_df['maturity_date'] < filter_date)]
            elif key == 'coupon_rate_min':
                # Fill NaN values in coupon_details
                filtered_df['coupon_details'] = filtered_df['coupon_details'].fillna('{}')

                # Extract coupon rate from coupon_details JSON
                filtered_df = filtered_df[
                    filtered_df['coupon_details'].apply(
                        lambda x: isinstance(x, dict) and
                        float(x.get('coupon_rate', 0).strip('%')) >= float(value.strip('%'))
                        if isinstance(x, dict) else False
                    )
                ]
            elif key == 'coupon_rate_max':
                # Fill NaN values in coupon_details
                filtered_df['coupon_details'] = filtered_df['coupon_details'].fillna('{}')

                # Extract coupon rate from coupon_details JSON
                filtered_df = filtered_df[
                    filtered_df['coupon_details'].apply(
                        lambda x: isinstance(x, dict) and
                        float(x.get('coupon_rate', 100).strip('%')) <= float(value.strip('%'))
                        if isinstance(x, dict) else False
                    )
                ]

        # Convert to list of dictionaries
        bonds_list = filtered_df.to_dict('records')

        # Clean up NaN values in each bond
        for bond in bonds_list:
            for key, value in list(bond.items()):
                if pd.isna(value):
                    bond[key] = ""

        return bonds_list

    def get_bonds_maturing_in_year(self, year: int) -> List[Dict]:
        """
        Get bonds maturing in a specific year.

        Args:
            year: The year to filter by

        Returns:
            List of dictionaries with bond details
        """
        if self.bond_details is None:
            return []

        # Fill NaN values in maturity_date column
        self.bond_details['maturity_date'] = self.bond_details['maturity_date'].fillna('')

        # Convert maturity_date to datetime
        self.bond_details['maturity_date'] = pd.to_datetime(
            self.bond_details['maturity_date'],
            format='%d-%m-%Y',
            errors='coerce'
        )

        # Filter by year, excluding rows with NaN dates
        matches = self.bond_details[
            ~self.bond_details['maturity_date'].isna() &
            (self.bond_details['maturity_date'].dt.year == year)
        ]

        if matches.empty:
            return []

        # Convert back to string format for consistency
        matches['maturity_date'] = matches['maturity_date'].dt.strftime('%d-%m-%Y')

        # Convert to list of dictionaries
        bonds_list = matches.to_dict('records')

        # Clean up NaN values in each bond
        for bond in bonds_list:
            for key, value in list(bond.items()):
                if pd.isna(value):
                    bond[key] = ""

        return bonds_list
