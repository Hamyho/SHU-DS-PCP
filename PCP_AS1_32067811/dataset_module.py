def get_data(filename=''):
    try:
        with open(filename, 'r') as file:
            transaction = {}
            for i in file:
                split_line = i.split(':')
                mini_diction = {'fraudulent': False if split_line[6].strip() == 'false' else True,
                                'x': float(split_line[4]),
                                'y': float(split_line[5]),
                                'amount': float(split_line[3]),
                                'description': split_line[2]}
                user_id = int(split_line[0])
                transaction_id = int(split_line[1])
                if user_id not in transaction.keys():
                    transaction[user_id] = {transaction_id: mini_diction}
                else:
                    transaction[user_id][transaction_id] = mini_diction
        return transaction
    except FileNotFoundError:
        print(f'Sorry, the file {filename} does not exist')
