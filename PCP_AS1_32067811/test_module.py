from distance_module import (
    distance_of_any_two_given_transaction_of_a_user,
    distance_of_any_two_given_transaction_of_any_user,
    integer_check,
    InvalidInputError,
)
import statistics_module as st


def main():
    statistic_functions = {
        1: {
            'statistic_name': 'Distance of a user',
            'option': distance_of_any_two_given_transaction_of_a_user
        },
        2: {
            'statistic_name': 'Distance of two user',
            'option': distance_of_any_two_given_transaction_of_any_user

        },
        3: {
            'statistic_name': 'Average of a user',
            'option': st.average_of_a_user
        },
        4: {
            'statistic_name': 'Average of all user',
            'option': st.average_of_all_user
        },
        5: {
            'statistic_name': 'Mode of a user',
            'option': st.mode_of_a_user
        },
        6: {
            'statistic_name': 'Mode of all user',
            'option': st.mode_of_all_user
        },
        7: {
            'statistic_name': 'Median of a user',
            'option': st.median_of_a_user
        },

        8: {
            'statistic_name': 'Median of all user',
            'option': st.median_of_all_user
        },

        9: {
            'statistic_name': 'Interquartile Range of a user',
            'option': st.interquartile_of_a_user
        },
        10: {
            'statistic_name': 'Interquartile Range of all user',
            'option': st.interquartile_of_all_user
        },
        11: {
            'statistic_name': 'Location Centroid',
            'option': st.location_centroid_a_user
        },
        12: {
            'statistic_name': 'Standard Deviation',
            'option': st.standard_deviation_of_a_user
        },
        13: {
            'statistic_name': 'Check fraudulent',
            'option': st.is_fraudulent
        },
        14: {
            'statistic_name': 'Abnormal Transaction',
            'option': st.abnormal_transaction_of_a_user
        },
        15: {
            'statistic_name': 'Z-score of a user',
            'option': st.z_score_of_a_user
        },
        16: {
            'statistic_name': 'Z-score of all user',
            'option': st.z_score_of_all_user
        },
        17: {
            'statistic_name': 'Frequency of transaction location',
            'option': st.frequencies_of_transaction
        },
        18: {
            'statistic_name': 'Outliers of location',
            'option': st.outlier_any_location_of_any_user
        },
        19: {
            'statistic_name': 'Nth quartile of a user',
            'option': st.nth_percentile_of_a_user
        },
        20: {
            'statistic_name': 'Nth quartile of all user',
            'option': st.nth_percentile_of_all_user
        }
    }

    message_menu = """Please choose one of these actions:
        1. Compute the distance between any two given transactions of a user.
        2. Compute the distance of transactions of any two users.
        3. Compute the average transactions of any user 
        4. Compute the average transactions of all users.
        5. Compute the mode of transactions of any user.
        6. Compute the mode of transactions of all users.
        7. Compute the median of all transactions of a user.
        8. Compute the median of all transactions of all users.
        9. Compute the interquartile range of any user's transactions.
        10. Compute the interquartile range of all user's transactions.
        11. Compute the location centroid of any user based on their transaction locations.
        12. Compute the standard deviation of any specific user’s transaction.
        13. Determine whether a transaction is fraudulent or not and provide details of such transactions.
        14. Find an abnormal transaction for any given user.
        15. Compute the Z-score of any user's transactions.
        16. Compute the Z-score for all users' transactions.
        17. Compute the frequencies of transactions at any given location.
        18. Find the outlier of any location and of any user.
        19. Compute the nth percentiles of transactions of any user 
        20. Compute the nth percentiles of transactions of all users.
        """
    while True:
        try:
            print(message_menu)
            choice = integer_check('Choice')
            if choice in statistic_functions:
                statistic_name = statistic_functions[choice]['statistic_name']
                statistic_fn = statistic_functions[choice]['option']
                print(f"{statistic_name} :", statistic_fn())
            else:
                print('Input choice wrong!!!')
        except InvalidInputError as e:
            print(str(e))
        except Exception as e:
            print(str(e))
        finally:
            continue_type = input('Do you want to exit: yes or no  ')
            if continue_type.lower() == 'yes' or continue_type.lower() == 'y':
                print('Good bye! ')
                break
            else:
                continue


main()
