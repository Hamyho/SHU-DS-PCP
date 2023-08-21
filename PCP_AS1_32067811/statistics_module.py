from distance_module import source_data, get_value, integer_check, InvalidInputError
import statistics as stats
import numpy as np


def get_valid_user_id(func):
    """
    Decorator to find valid user Id input in dataset
    :param func:
    :return: function with a valid user Id parameter
    """

    def wrapper():
        try:
            user_id = get_value('User_id', source_data)
            return func(user_id)
        except InvalidInputError as e:
            print(str(e))

    return wrapper


def get_list_amount_of_a_user(user_id):
    """
    Get all transactions of a user according to a given user id
    :param user_id: a given user id
    :return: array transactions of the user
    """
    return np.array([v.get('amount') for k, v in source_data[user_id].items()])


def get_list_amount_of_all_user():
    """
    Get all transactions
    :return: array of amounts of all transactions
    """
    return np.array([value['amount'] for user_id in source_data for transaction, value in source_data[user_id].items()])


@get_valid_user_id
def average_of_a_user(user_id):
    """
    Calculate the average of transactions according to a given user id
    :param user_id: a given user id
    :return: the average of  the transactions
    """
    amounts = get_list_amount_of_a_user(user_id)
    return amounts.mean()


def average_of_all_user():
    """
    Calculate the average of transactions of all users
    :return: list average of transactions of every user
    """

    return get_list_amount_of_all_user().mean()


@get_valid_user_id
def mode_of_a_user(user_id):
    """
    Calculate the mode of transactions of any given user
    :param user_id: a given user id
    :return: mode of the transaction
    """
    amounts = get_list_amount_of_a_user(user_id)
    return stats.mode(amounts)


def mode_of_all_user():
    """
    Calculate the mode of transactions of all user
    :return: list mode of the user's transactions.
    """

    return stats.mode(get_list_amount_of_all_user())


@get_valid_user_id
def median_of_a_user(user_id):
    """
    Calculate the median of transactions of any given user
    :param user_id: a given user id
    :return: median of the transaction
    """

    amounts = get_list_amount_of_a_user(user_id)
    return np.median(amounts)


def median_of_all_user():
    """
    Calculate the median of transactions of all user
    :return: list median of the user's transactions.
    """

    return np.median(get_list_amount_of_all_user())


@get_valid_user_id
def interquartile_of_a_user(user_id):
    """
     Calculate the interquartile of transactions of any given user
    :param user_id: a given user id
    :return: interquartile of the transaction
    """
    amounts = get_list_amount_of_a_user(user_id)
    q_3, q_1 = np.percentile(amounts, [75, 25])
    iqr = q_3 - q_1
    return iqr


def interquartile_of_all_user():
    """
    Calculate the interquartile of transactions of all user
    :return: list interquartile of the user's transactions.
    """
    amounts = get_list_amount_of_all_user()
    q_3, q_1 = np.percentile(amounts, [75, 25])
    iqr = q_3 - q_1
    return iqr


@get_valid_user_id
def location_centroid_a_user(user_id):
    """
    Calculate the location centroid of any user based on their transaction locations.
    :param user_id: a given user id
    :return: x, y point of location centroid
    """

    list_of_location = np.array([[v.get('x'), v.get('y')] for k, v in source_data[user_id].items()])
    return list_of_location.mean(axis=0)


@get_valid_user_id
def standard_deviation_of_a_user(user_id):
    """
    Calculate the standard deviation of any user.
    :param user_id: a given user id
    :return:  standard deviation of the user
    """
    amounts = get_list_amount_of_a_user(user_id)
    return amounts.std()


def standard_deviation_of_all_user():
    """
    Calculate the standard deviation of all user.
    :return:  standard deviation of all user
    """
    amounts = get_list_amount_of_all_user()
    return amounts.std()


def get_valid_integer_type_of_transaction_id(func):
    def wrapper():
        try:
            nth = integer_check('transaction_id')
            return func(nth)
        except InvalidInputError as e:
            print(str(e))

    return wrapper


@get_valid_integer_type_of_transaction_id
def is_fraudulent(transaction_id):
    """
    Check a transaction is fraudulent or not
    :param transaction_id: a given transaction id
    :return: list of status of fraudulent and the transaction information.
    """

    list_of_fraudulent = list(source_data.values())
    for index in list_of_fraudulent:
        for key, value in index.items():
            if key == transaction_id:
                return value['fraudulent'], value
    raise InvalidInputError('Transaction Id does not exist')


@get_valid_user_id
def abnormal_transaction_of_a_user(user_id):
    """
    Calculate abnormal transactions of a user
    :param user_id: a given user id
    :return: list of abnormal transaction
    """
    amounts = np.array([[int(transaction_id), v.get('amount')] for transaction_id, v in source_data[user_id].items()])
    q_3_x, q_1_x = np.percentile(amounts[:, 1], [75, 25])
    iqr_x = q_3_x - q_1_x
    lower_x = q_1_x - 1.5 * iqr_x
    upper_x = q_3_x + 1.5 * iqr_x
    return [(int(transaction_id), source_data[user_id][transaction_id]) for transaction_id, v in amounts if
            v <= lower_x or v >= upper_x]


@get_valid_user_id
def z_score_of_a_user(user_id):
    """
    Calculate z_score  of a user
    :param user_id: a given user id
    :return: list of z_score
    """
    amounts = get_list_amount_of_a_user(user_id)
    mean = np.mean(amounts)
    std = np.std(amounts)
    return [[x, (x - mean) / std] for x in get_list_amount_of_a_user(user_id)]


def z_score_of_all_user():
    """
    Calculate z_score of transactions of all user
    :return: list z_score of the user's transactions.
    """

    mean = average_of_all_user()
    std = standard_deviation_of_all_user()
    return [[x, (x - mean) / std] for x in get_list_amount_of_all_user()]


def get_location(func):
    """
    Input value and check input exception
    :param func:
    :return: function with 2 correct parameters
    """

    def wrapper():
        try:
            location_x = input('Please input x point: ')
            if not location_x.replace('.', '', 1).isdigit():
                raise InvalidInputError(f'Invalid x coordinate {location_x}. Please input a float number.')
            location_y = input('Please input y point: ')
            if not location_y.replace('.', '', 1).isdigit():
                raise InvalidInputError(f'Invalid y coordinate {location_y}. Please input a float number.')
            return func(float(location_x), float(location_y))
        except InvalidInputError as e:
            print(str(e))

    return wrapper


@get_location
def frequencies_of_transaction(location_x, location_y):
    """
    Computes those frequencies of transactions at any given location.
    :param location_x: a given x point
    :param location_y: a given y point
    :return: those frequencies of transactions based x, y  given point.
    """
    freq = 0
    for key_1, item in source_data.items():
        for key_2, item2 in item.items():
            if item2['x'] == location_x and item2['y'] == location_y:
                freq += 1
    return freq


@get_valid_user_id
def outlier_any_location_of_any_user(user_id):
    """
    Computes outlier of any location and of any user.
    :param user_id: a given user id
    :return: list of outlier
    """
    location = np.array([[v.get('x'), v.get('y')] for k, v in source_data[user_id].items()])
    x = location[:, 0:1]
    y = location[:, 1:2]

    # calculate Q1 and Q3 of location of x coordinates

    q_3_x, q_1_x = np.percentile(x, [75, 25])
    iqr_x = q_3_x - q_1_x
    lower_x = q_1_x - 1.5 * iqr_x
    upper_x = q_3_x + 1.5 * iqr_x
    # calculate Q1 and Q3 of location of x coordinates
    q_3_y, q_1_y = np.percentile(y, [75, 25])
    iqr_y = q_3_y - q_1_y
    lower_y = q_1_y - 1.5 * iqr_y
    upper_y = q_3_y + 1.5 * iqr_y
    outlier = [[l_x, l_y] for l_x, l_y in location if
               (l_x < lower_x or l_x > upper_x or l_y < lower_y or l_y > upper_y)]
    return outlier


def get_user_id_and_nth(func):
    """
    Decorator to check user id and nth value valid or not
    :param func:
    :return: function with two valid parameters
    """

    def wrapper():
        try:
            user_id = integer_check('user_id')
            nth = integer_check('nth')
            return func(user_id, nth)
        except InvalidInputError as e:
            print(str(e))

    return wrapper


@get_user_id_and_nth
def nth_percentile_of_a_user(user_id, n):
    """
    Nth percentiles of transactions of any user.
    :param user_id: a given user id
    :param n: nth percentiles
    :return: the nth percentiles of transactions of the user
    """

    amount = get_list_amount_of_a_user(user_id)
    return np.percentile(amount, n)


def get_valid_nth(func):
    def wrapper():
        try:
            nth = integer_check('nth')
            return func(nth)
        except InvalidInputError as e:
            print(str(e))

    return wrapper


@get_valid_nth
def nth_percentile_of_all_user(n):
    """
    Nth percentiles of transactions of all user.
    :param n: nth percentiles
    :return: the nth percentiles of transactions of the user
    """
    amount = get_list_amount_of_all_user()
    return np.percentile(amount, n)
