# New trend_analysis_stats function content
        total_contacts = len(df)

        # Analyze success rate (y = yes means customer subscribed to term deposit)
        subscribed = df[df['y'] == 'yes']
        total_subscribed = len(subscribed)
        subscription_rate = (total_subscribed / total_contacts * 100) if total_contacts > 0 else 0

        # Group by month to find peak month
        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                      'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        contacts_by_month = df.groupby('month').size()
        peak_month = contacts_by_month.idxmax() if len(contacts_by_month) > 0 else 'may'
        avg_contacts_per_month = contacts_by_month.mean() if len(contacts_by_month) > 0 else 0

        # Bank Term Deposit Trends - group by month
        months_in_data = sorted(df['month'].unique(), key=lambda x: month_order.get(x, 0))

        # Get last month and second to last month data
        if len(months_in_data) >= 2:
            last_month = months_in_data[-1]
            second_last_month = months_in_data[-2]

            deposits_last_month = len(df[df['month'] == last_month])
            deposits_second_last = len(df[df['month'] == second_last_month])

            subscribed_last_month = len(df[(df['month'] == last_month) & (df['y'] == 'yes')])
            subscribed_second_last = len(df[(df['month'] == second_last_month) & (df['y'] == 'yes')])
        else:
            last_month = months_in_data[0] if len(months_in_data) > 0 else 'may'
            deposits_last_month = len(df[df['month'] == last_month])
            deposits_second_last = deposits_last_month
            subscribed_last_month = len(df[(df['month'] == last_month) & (df['y'] == 'yes')])
            subscribed_second_last = subscribed_last_month

        # Calculate volumes by periods
        total_months = len(months_in_data)
        volume_this_month = deposits_last_month
        volume_last_month = deposits_second_last

        # Approximate based on available data
        volume_last_3_months = len(df[df['month'].isin(months_in_data[-3:])]) if total_months >= 3 else total_contacts
        volume_last_6_months = len(df[df['month'].isin(months_in_data[-6:])]) if total_months >= 6 else total_contacts
        volume_ytd = total_contacts  # Year to date = all data

        # Success Rate Distribution (based on campaign effectiveness)
        # Segment by job type for success analysis
        job_success = df.groupby('job').apply(lambda x: (x['y'] == 'yes').sum() / len(x) * 100 if len(x) > 0 else 0)

        high_success_jobs = job_success[job_success > 40].index.tolist()
        medium_success_jobs = job_success[(job_success >= 20) & (job_success <= 40)].index.tolist()
        low_success_jobs = job_success[job_success < 20].index.tolist()

        high_success_count = len(df[df['job'].isin(high_success_jobs)])
        medium_success_count = len(df[df['job'].isin(medium_success_jobs)])
        low_success_count = len(df[df['job'].isin(low_success_jobs)])

        total_analyzed = high_success_count + medium_success_count + low_success_count
        high_success_pct = (high_success_count / total_analyzed * 100) if total_analyzed > 0 else 0
        medium_success_pct = (medium_success_count / total_analyzed * 100) if total_analyzed > 0 else 0
        low_success_pct = (low_success_count / total_analyzed * 100) if total_analyzed > 0 else 0

        # Calculate percentage changes
        change_month = ((volume_this_month - volume_last_month) / volume_last_month * 100) if volume_last_month > 0 else 0
        avg_per_month_3m = volume_last_3_months / min(3, total_months) if volume_last_3_months > 0 else 0
        avg_per_month_6m = volume_last_6_months / min(6, total_months) if volume_last_6_months > 0 else 0

        change_3m = ((avg_per_month_3m - volume_last_month) / volume_last_month * 100) if volume_last_month > 0 else 0
        change_6m = ((avg_per_month_6m - avg_per_month_3m) / avg_per_month_3m * 100) if avg_per_month_3m > 0 else 0

        # Calculate subscription trend
        subscription_trend = ((subscribed_last_month - subscribed_second_last) / subscribed_second_last * 100) if subscribed_second_last > 0 else 0

        return jsonify({
            'success': True,
            'total_contacts': total_contacts,
            'total_subscribed': total_subscribed,
            'subscription_rate': round(subscription_rate, 1),
            'campaign_trends': {
                'peak_month': peak_month.capitalize(),
                'avg_per_month': round(avg_contacts_per_month, 0),
                'subscription_trend': f"{subscription_trend:+.0f}%"
            },
            'deposit_trends': {
                'this_month': volume_this_month,
                'last_month': volume_last_month,
                'last_3_months': volume_last_3_months,
                'last_6_months': volume_last_6_months,
                'ytd': volume_ytd,
                'changes': {
                    'this_month': f"{change_month:+.0f}%",
                    'last_3_months': f"{change_3m:+.0f}%",
                    'last_6_months': f"{change_6m:+.0f}%",
                    'ytd': f"{subscription_trend:+.0f}%"
                }
            },
            'success_distribution': {
                'high_success': {
                    'count': high_success_count,
                    'percentage': round(high_success_pct, 0)
                },
                'medium_success': {
                    'count': medium_success_count,
                    'percentage': round(medium_success_pct, 0)
                },
                'low_success': {
                    'count': low_success_count,
                    'percentage': round(low_success_pct, 0)
                }
            }
        })
