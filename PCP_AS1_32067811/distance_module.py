from dataset_module import get_data
from math import sqrt

source_data = get_data('Transaction.txt')


class InvalidInputError(Exception):
    pass


def integer_check(index):
    """
    Function to input value
    :param index: any given value
    :return: an integer value
    """
    value = input(f'Please insert {index}: ')
    if not value.isdigit():
        raise InvalidInputError(f'Invalid {index}. Please input a number.')
    return int(value)


def get_value(index, data):
    """
    Function to find input value in dataset
    :param index: input value
    :param data: dataset
    :return: a correct value
    """
    value = integer_check(index)
    if value not in data:
        raise InvalidInputError(f'{index} {value} does not exist')
    return value


def get_two_transaction_id_and_a_user_id(func):
    """
    This is decorator to check two transaction ID of two user id are valid or not in source data
    :param func:
    :return: function with correct parameters
    """
    
    def wrapper():
        try:
            user_id = get_value('User_id', source_data)
            trans_1_id = get_value('Transaction_id_1', source_data[user_id])
            trans_2_id = get_value('Transaction_id_2', source_data[user_id])
            return func(user_id, trans_1_id, trans_2_id)
        except InvalidInputError as e:
            print(str(e))

    return wrapper


def get_two_transaction_id_and_two_user_id(func):
    """
    This is decorator to check two transaction ID and two user id are valid or not in source data
    """
    def wrapper():
        try:
            user_id_1 = get_value('User_id_1', source_data)
            trans_1_id = get_value('Transaction_id_1', source_data[user_id_1])
            user_id_2 = get_value('User_id_2', source_data)
            trans_2_id = get_value('Transaction_id_2', source_data[user_id_2])
            return func(user_id_1, trans_1_id, user_id_2, trans_2_id)
        except InvalidInputError as e:
            print(str(e))
    return wrapper


@get_two_transaction_id_and_a_user_id
def distance_of_any_two_given_transaction_of_a_user(user_id, trans_1_id, trans_2_id):
    """
    Compute distance of any two given transaction of a user
    :param user_id: a given user ID. e.g. 21
    :param trans_1_id: a first transaction ID of the user ID. e.g. 500000
    :param trans_2_id:  a second transaction ID of the user ID. e.g. 500001
    :return: a float value of distance
    """
    x_1 = source_data[user_id][trans_1_id]['x']
    y_1 = source_data[user_id][trans_1_id]['y']
    x_2 = source_data[user_id][trans_2_id]['x']
    y_2 = source_data[user_id][trans_2_id]['y']
    print(f'Distance of any two given transaction of a user_id {user_id}:')
    print(f'First is {source_data[user_id][trans_1_id]}')
    print(f'Second is {source_data[user_id][trans_2_id]}')
    return sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


@get_two_transaction_id_and_two_user_id
def distance_of_any_two_given_transaction_of_any_user(user_id_1, trans_1_id, user_id_2, trans_2_id):
    """
    Compute distance of any two given transaction of two user
    :param user_id_1: a first user ID. e.g. 21
    :param trans_1_id: a first transaction ID of the first user ID. e.g. 500000
    :param user_id_2: a second user ID. e.g. 22
    :param trans_2_id:  a first transaction ID of the second user ID. e.g. 500200
    :return: a float value of distance
    """
    x_1 = source_data[user_id_1][trans_1_id]['x']
    y_1 = source_data[user_id_1][trans_1_id]['y']
    x_2 = source_data[user_id_2][trans_2_id]['x']
    y_2 = source_data[user_id_2][trans_2_id]['y']
    print(f'Information of any two given transaction of a user_id {user_id_1} and transaction id {trans_1_id} is:')
    print(f'First is {source_data[user_id_1][trans_1_id]}')
    print(f'Information of any two given transaction of a user_id {user_id_2} and transaction id {trans_2_id} is:')
    print(f'Second is {source_data[user_id_2][trans_2_id]}')
    return sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


