import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


def detect_dataset_type(df):
    """Detect dataset type based on columns"""
    if 'card_number' in df.columns and 'cvv' in df.columns:
        return 'cards'
    elif 'merchant_id' in df.columns and 'use_chip' in df.columns:
        return 'transactions'
    elif 'credit_score' in df.columns and 'yearly_income' in df.columns and 'latitude' in df.columns:
        return 'users'
    else:
        return 'unknown'


def check_transactions_quality(df, dataset_name):
    """Data quality checks for transactions dataset"""
    results = {
        'dataset_name': dataset_name,
        'dataset_type': 'transactions',
        'total_records': len(df),
        'total_columns': len(df.columns),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'completeness': {},
        'uniqueness': {},
        'accuracy': {},
        'consistency': {},
        'statistics': {},
        'issues': []
    }
    
    # Completeness (exclude 'errors' column if present)
    columns_to_check = [col for col in df.columns if col not in ['errors']]
    completeness = df[columns_to_check].notnull().mean()
    for col, score in completeness.items():
        status = "GREEN" if score >= 0.95 else ("YELLOW" if score >= 0.85 else "RED")
        missing_count = df[col].isnull().sum()
        results['completeness'][col] = {
            'score': float(score),
            'status': status,
            'missing_count': int(missing_count)
        }
        if missing_count > 0:
            results['issues'].append(f"{missing_count:,} missing values in {col}")
    
    # Uniqueness
    unique_id = df['id'].nunique() / len(df)
    status = "GREEN" if unique_id >= 0.95 else ("YELLOW" if unique_id >= 0.85 else "RED")
    results['uniqueness']['transaction_id'] = {
        'score': float(unique_id),
        'status': status
    }
    
    dup_ids = df[df.duplicated(subset=['id'], keep=False)]
    if len(dup_ids) > 0:
        results['issues'].append(f"{len(dup_ids):,} duplicate transaction IDs")
    
    # Convert amount to numeric if it exists
    if 'amount' in df.columns:
        df['amount_numeric'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Accuracy - Amount validation
    if 'amount' in df.columns:
        invalid_amount = df[(df['amount_numeric'] <= 0) | (df['amount_numeric'] > 100000)]
        amount_score = 1 - (len(invalid_amount) / len(df))
        status = "GREEN" if amount_score >= 0.95 else ("YELLOW" if amount_score >= 0.85 else "RED")
        results['accuracy']['amount'] = {
            'score': float(amount_score),
            'status': status,
            'invalid_count': int(len(invalid_amount))
        }
        if len(invalid_amount) > 0:
            results['issues'].append(f"{len(invalid_amount):,} transactions with invalid amounts")
    
    # Accuracy - use_chip validation (should be 'Chip Transaction' or 'Swipe Transaction')
    if 'use_chip' in df.columns:
        valid_chip_values = ['Chip Transaction', 'Swipe Transaction', 'Online Transaction']
        invalid_chip = df[~df['use_chip'].astype(str).isin(valid_chip_values)]
        chip_score = 1 - (len(invalid_chip) / len(df))
        status = "GREEN" if chip_score >= 0.95 else ("YELLOW" if chip_score >= 0.85 else "RED")
        results['accuracy']['use_chip'] = {
            'score': float(chip_score),
            'status': status,
            'invalid_count': int(len(invalid_chip))
        }
        if len(invalid_chip) > 0:
            results['issues'].append(f"{len(invalid_chip):,} invalid chip transaction types")
    
    # Accuracy - Merchant validation
    if 'merchant_id' in df.columns:
        invalid_merchant = df[df['merchant_id'].isnull()]
        merchant_score = 1 - (len(invalid_merchant) / len(df))
        status = "GREEN" if merchant_score >= 0.95 else ("YELLOW" if merchant_score >= 0.85 else "RED")
        results['accuracy']['merchant_id'] = {
            'score': float(merchant_score),
            'status': status,
            'invalid_count': int(len(invalid_merchant))
        }
    
    # Accuracy - Merchant city validation
    if 'merchant_city' in df.columns:
        invalid_city = df[df['merchant_city'].isnull() | (df['merchant_city'].astype(str).str.len() < 2)]
        city_score = 1 - (len(invalid_city) / len(df))
        status = "GREEN" if city_score >= 0.95 else ("YELLOW" if city_score >= 0.85 else "RED")
        results['accuracy']['merchant_city'] = {
            'score': float(city_score),
            'status': status,
            'invalid_count': int(len(invalid_city))
        }
    
    # Accuracy - MCC code validation (4-digit codes)
    if 'mcc' in df.columns:
        invalid_mcc = df[~df['mcc'].astype(str).str.match(r'^\d{4}$', na=False)]
        mcc_score = 1 - (len(invalid_mcc) / len(df))
        status = "GREEN" if mcc_score >= 0.95 else ("YELLOW" if mcc_score >= 0.85 else "RED")
        results['accuracy']['mcc'] = {
            'score': float(mcc_score),
            'status': status,
            'invalid_count': int(len(invalid_mcc))
        }
        if len(invalid_mcc) > 0:
            results['issues'].append(f"{len(invalid_mcc):,} invalid MCC codes")
    
    # Consistency - Check if errors field matches is_fraud
    if 'errors' in df.columns and 'is_fraud' in df.columns:
        # Convert is_fraud to numeric if it's a string
        df['is_fraud_numeric'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
        
        # Transactions with errors should potentially be fraud
        has_error = df['errors'].notna() & (df['errors'].astype(str) != '') & (df['errors'].astype(str) != 'nan')
        consistency_issues = df[has_error & (df['is_fraud_numeric'] == 0)]
        if len(consistency_issues) > 0:
            consistency_score = 1 - (len(consistency_issues) / len(df))
            status = "GREEN" if consistency_score >= 0.95 else ("YELLOW" if consistency_score >= 0.85 else "RED")
            results['consistency']['error_fraud_match'] = {
                'score': float(consistency_score),
                'status': status,
                'invalid_count': int(len(consistency_issues))
            }
            results['issues'].append(f"{len(consistency_issues):,} transactions with errors but not marked as fraud")
    
    # Consistency - Card/User match
    if 'card_number' in df.columns and 'user_id' in df.columns:
        # Check if same card is used by multiple users (potential fraud indicator)
        card_user_combos = df.groupby('card_number')['user_id'].nunique()
        multi_user_cards = card_user_combos[card_user_combos > 1]
        if len(multi_user_cards) > 0:
            affected_transactions = df[df['card_number'].isin(multi_user_cards.index)]
            consistency_score = 1 - (len(affected_transactions) / len(df))
            status = "GREEN" if consistency_score >= 0.95 else ("YELLOW" if consistency_score >= 0.85 else "RED")
            results['consistency']['card_user_match'] = {
                'score': float(consistency_score),
                'status': status,
                'invalid_count': int(len(affected_transactions))
            }
            results['issues'].append(f"{len(multi_user_cards):,} cards used by multiple users ({len(affected_transactions):,} transactions)")
    
    # Statistics
    stats = {
        'total_transactions': int(len(df))
    }
    
    if 'amount' in df.columns and 'amount_numeric' in df.columns:
        stats['total_amount'] = float(df['amount_numeric'].sum())
        stats['avg_transaction'] = float(df['amount_numeric'].mean())
        stats['max_transaction'] = float(df['amount_numeric'].max())
        stats['min_transaction'] = float(df['amount_numeric'].min())
    
    if 'is_fraud' in df.columns:
        df['is_fraud_numeric'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
        fraud_count = int(df['is_fraud_numeric'].sum())
        stats['fraud_transactions'] = fraud_count
        stats['fraud_rate'] = float(fraud_count / len(df) * 100) if len(df) > 0 else 0.0
    
    if 'use_chip' in df.columns:
        stats['transaction_type_dist'] = df['use_chip'].value_counts().to_dict()
    
    if 'merchant_city' in df.columns:
        top_cities = df['merchant_city'].value_counts().head(5).to_dict()
        stats['top_cities'] = top_cities
    
    results['statistics'] = stats
    
    return results



def check_users_quality(df, dataset_name):
    """Data quality checks for users/customers dataset"""
    results = {
        'dataset_name': dataset_name,
        'dataset_type': 'users',
        'total_records': len(df),
        'total_columns': len(df.columns),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'completeness': {},
        'uniqueness': {},
        'accuracy': {},
        'consistency': {},
        'statistics': {},
        'issues': []
    }
    
    # Completeness
    completeness = df.notnull().mean()
    for col, score in completeness.items():
        status = "GREEN" if score >= 0.95 else ("YELLOW" if score >= 0.85 else "RED")
        missing_count = df[col].isnull().sum()
        results['completeness'][col] = {
            'score': float(score),
            'status': status,
            'missing_count': int(missing_count)
        }
        if missing_count > 0:
            results['issues'].append(f"{missing_count:,} missing values in {col}")
    
    # Uniqueness
    unique_id = df['id'].nunique() / len(df)
    status = "GREEN" if unique_id >= 0.95 else ("YELLOW" if unique_id >= 0.85 else "RED")
    results['uniqueness']['user_id'] = {
        'score': float(unique_id),
        'status': status
    }
    
    dup_ids = df[df.duplicated(subset=['id'], keep=False)]
    if len(dup_ids) > 0:
        results['issues'].append(f"{len(dup_ids):,} duplicate user IDs")
    
    dup_address = df[df.duplicated(subset=['address'], keep=False)]
    if len(dup_address) > 0:
        dup_addr_score = 1 - (len(dup_address) / len(df))
        status = "GREEN" if dup_addr_score >= 0.95 else ("YELLOW" if dup_addr_score >= 0.85 else "RED")
        results['uniqueness']['address'] = {
            'score': float(dup_addr_score),
            'status': status
        }
        results['issues'].append(f"{len(dup_address):,} duplicate addresses")
    
    # Accuracy checks
    invalid_age = df[(df['current_age'] < 18) | (df['current_age'] > 100)]
    age_score = 1 - (len(invalid_age) / len(df))
    status = "GREEN" if age_score >= 0.95 else ("YELLOW" if age_score >= 0.85 else "RED")
    results['accuracy']['age'] = {
        'score': float(age_score),
        'status': status,
        'invalid_count': int(len(invalid_age))
    }
    if len(invalid_age) > 0:
        results['issues'].append(f"{len(invalid_age):,} users with invalid age (not 18-100)")
    
    valid_genders = ['Male', 'Female', 'M', 'F', 'Other', 'Non-binary']
    invalid_gender = df[~df['gender'].isin(valid_genders)]
    gender_score = 1 - (len(invalid_gender) / len(df))
    status = "GREEN" if gender_score >= 0.95 else ("YELLOW" if gender_score >= 0.85 else "RED")
    results['accuracy']['gender'] = {
        'score': float(gender_score),
        'status': status,
        'invalid_count': int(len(invalid_gender))
    }
    if len(invalid_gender) > 0:
        results['issues'].append(f"{len(invalid_gender):,} invalid gender values")
    
    invalid_credit = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
    credit_score = 1 - (len(invalid_credit) / len(df))
    status = "GREEN" if credit_score >= 0.95 else ("YELLOW" if credit_score >= 0.85 else "RED")
    results['accuracy']['credit_score'] = {
        'score': float(credit_score),
        'status': status,
        'invalid_count': int(len(invalid_credit))
    }
    if len(invalid_credit) > 0:
        results['issues'].append(f"{len(invalid_credit):,} invalid credit scores")
    
    invalid_lat = df[(df['latitude'] < 24) | (df['latitude'] > 50)]
    invalid_lon = df[(df['longitude'] < -125) | (df['longitude'] > -66)]
    geo_invalid = len(invalid_lat) + len(invalid_lon)
    geo_score = 1 - (geo_invalid / (len(df) * 2))
    status = "GREEN" if geo_score >= 0.95 else ("YELLOW" if geo_score >= 0.85 else "RED")
    results['accuracy']['geo_location'] = {
        'score': float(geo_score),
        'status': status,
        'invalid_count': int(geo_invalid)
    }
    if geo_invalid > 0:
        results['issues'].append(f"{geo_invalid:,} invalid coordinates")
    
    # Consistency checks
    invalid_retirement = df[df['retirement_age'] <= df['current_age']]
    retirement_score = 1 - (len(invalid_retirement) / len(df))
    status = "GREEN" if retirement_score >= 0.95 else ("YELLOW" if retirement_score >= 0.85 else "RED")
    results['consistency']['retirement_age'] = {
        'score': float(retirement_score),
        'status': status,
        'invalid_count': int(len(invalid_retirement))
    }
    if len(invalid_retirement) > 0:
        results['issues'].append(f"{len(invalid_retirement):,} users with invalid retirement age")
    
    current_year = 2020
    df['calculated_age'] = current_year - df['birth_year']
    age_mismatch = df[abs(df['calculated_age'] - df['current_age']) > 1]
    age_consistency = 1 - (len(age_mismatch) / len(df))
    status = "GREEN" if age_consistency >= 0.95 else ("YELLOW" if age_consistency >= 0.85 else "RED")
    results['consistency']['birth_year'] = {
        'score': float(age_consistency),
        'status': status,
        'invalid_count': int(len(age_mismatch))
    }
    if len(age_mismatch) > 0:
        results['issues'].append(f"{len(age_mismatch):,} age/birth year mismatches")
    
    # Statistics
    df['income_numeric'] = pd.to_numeric(df['yearly_income'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['debt_numeric'] = pd.to_numeric(df['total_debt'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    results['statistics'] = {
        'total_users': int(len(df)),
        'age_range': f"{int(df['current_age'].min())}-{int(df['current_age'].max())}",
        'avg_credit_score': float(df['credit_score'].mean()),
        'avg_yearly_income': float(df['income_numeric'].mean()),
        'avg_total_debt': float(df['debt_numeric'].mean()),
        'avg_credit_cards': float(df['num_credit_cards'].mean()),
        'gender_distribution': df['gender'].value_counts().to_dict()
    }
    
    return results


def check_cards_quality(df, dataset_name):
    """Data quality checks for cards dataset"""
    results = {
        'dataset_name': dataset_name,
        'dataset_type': 'cards',
        'total_records': len(df),
        'total_columns': len(df.columns),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'completeness': {},
        'uniqueness': {},
        'accuracy': {},
        'timeliness': {},
        'statistics': {},
        'issues': []
    }
    
    # Completeness
    completeness = df.notnull().mean()
    for col, score in completeness.items():
        status = "GREEN" if score >= 0.95 else ("YELLOW" if score >= 0.85 else "RED")
        missing_count = df[col].isnull().sum()
        results['completeness'][col] = {
            'score': float(score),
            'status': status,
            'missing_count': int(missing_count)
        }
        if missing_count > 0:
            results['issues'].append(f"{missing_count:,} missing values in {col}")
    
    # Uniqueness
    unique_card = df['card_number'].nunique() / len(df)
    unique_id = df['id'].nunique() / len(df)
    
    status_card = "GREEN" if unique_card >= 0.95 else ("YELLOW" if unique_card >= 0.85 else "RED")
    status_id = "GREEN" if unique_id >= 0.95 else ("YELLOW" if unique_id >= 0.85 else "RED")
    
    results['uniqueness']['card_number'] = {
        'score': float(unique_card),
        'status': status_card
    }
    results['uniqueness']['id'] = {
        'score': float(unique_id),
        'status': status_id
    }
    
    # Accuracy
    valid_brands = ['Visa', 'Mastercard', 'Discover', 'JCB', 'Amex', 'American Express']
    brand_score = df['card_brand'].isin(valid_brands).mean()
    status = "GREEN" if brand_score >= 0.95 else ("YELLOW" if brand_score >= 0.85 else "RED")
    results['accuracy']['card_brand'] = {
        'score': float(brand_score),
        'status': status
    }
    
    cvv_score = df['cvv'].astype(str).str.match(r'^\d{3,4}$', na=False).mean()
    status = "GREEN" if cvv_score >= 0.95 else ("YELLOW" if cvv_score >= 0.85 else "RED")
    results['accuracy']['cvv'] = {
        'score': float(cvv_score),
        'status': status
    }
    if cvv_score < 0.95:
        results['issues'].append(f"{int((1-cvv_score)*len(df)):,} invalid CVV formats")
    
    # Timeliness
    def is_expired(expire_str):
        try:
            month, year = map(int, str(expire_str).split('/'))
            year = 2000 + year if year < 100 else year
            return year < 2020 or (year == 2020 and month < 11)
        except:
            return True
    
    expired_count = df['expires'].apply(is_expired).sum()
    timeliness_score = 1 - (expired_count / len(df))
    status = "GREEN" if timeliness_score >= 0.95 else ("YELLOW" if timeliness_score >= 0.85 else "RED")
    results['timeliness']['expired'] = {
        'score': float(timeliness_score),
        'status': status,
        'expired_count': int(expired_count)
    }
    if expired_count > 0:
        results['issues'].append(f"{expired_count:,} expired cards")
    
    results['statistics'] = {
        'total_cards': int(len(df)),
        'brand_distribution': df['card_brand'].value_counts().to_dict()
    }
    
    return results


def check_data_quality(file_path):
    """Main function to check any CSV file"""
    if not os.path.exists(file_path):
        return {
            'error': f"File not found: {file_path}",
            'dataset_name': file_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    try:
        df = pd.read_csv(file_path)
        dataset_type = detect_dataset_type(df)
        
        if dataset_type == 'cards':
            return check_cards_quality(df, file_path)
        elif dataset_type == 'users':
            return check_users_quality(df, file_path)
        elif dataset_type == 'transactions':
            return check_transactions_quality(df, file_path)
        else:
            return {
                'error': f"Unknown dataset type. Columns: {list(df.columns)}",
                'dataset_name': file_path,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    except Exception as e:
        return {
            'error': str(e),
            'dataset_name': file_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("=" * 70)
    print("🔍 DATA QUALITY CHECKER - Web Export")
    print("=" * 70)
    
    datasets = [
        'cards_data.csv',
        'users_data.csv',
        'transactions_data.csv',
    ]
    
    all_results = []
    
    for dataset_file in datasets:
        print(f"\n✓ Processing: {dataset_file}")
        result = check_data_quality(dataset_file)
        all_results.append(result)
    
    # Function to clean NaN values for JSON
    def clean_for_json(obj):
        """Recursively replace NaN, inf, -inf with None for valid JSON"""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        else:
            return obj
    
    # Clean the results
    all_results = clean_for_json(all_results)
    
    # Save results as JSON
    output_file = 'quality_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 70)


