import sqlite3
import model


class DB:
    def __init__(self):
        self._conn = sqlite3.connect('lms.db', check_same_thread=False)
        self._c = self._conn.cursor()

    # run this method only once
    def create_table_user(self):
        self._c.execute("""CREATE TABLE users(
            id integer PRIMARY KEY,
            name text,
            email text,
            password text
        )""")
        self._conn.commit()

    def login(self, email, password):
        statement = f"SELECT * FROM users WHERE email='{email}' AND password = '{password}';"
        self._c.execute(statement)
        if not self._c.fetchone():  # An empty result evaluates to False.
            print("Login failed")
            return False
        else:
            print("Welcome")
            return True

    def get_user_by_email(self, email):
        statement = f"SELECT * FROM users WHERE email='{email}'"
        self._c.execute(statement)
        users = self._c.fetchall()
        returnUserList = []
        for i in range(0, len(users)):
            returnUserList.append(model.User(users[i][0], users[i][1], users[i][2], users[i][3]))
        return returnUserList

    def select_user(self):
        self._c.execute("SELECT * FROM users")
        users = self._c.fetchall()
        returnUserList = []
        for i in range(0, len(users)):
            returnUserList.append(model.User(users[i][0], users[i][1], users[i][2], users[i][3]))
        return returnUserList

    def save_user(self, name, email, password):
        sql = "INSERT INTO users (name, email, password) " \
              "VALUES (?, ?, ?)"
        val = (name, email, password)
        self._c.execute(sql, val)
        self._conn.commit()

    def update_user(self, id, name, email, password):
        sql = "UPDATE users SET name = ?, email = ?, password = ? WHERE id = ?"
        val = (name, email, password, id)
        self._c.execute(sql, val)
        self._conn.commit()

    def delete_user(self, id):
        sql = 'DELETE FROM users WHERE id = ?'
        val = (id)
        self._c.execute(sql, val)
        self._conn.commit()

# db = DB()
# print(db.get_user_by_email('sankamadushanka78@gmail.com')[0].get_name())
# db.create_table_user()
# db.save_user('Roshan Withanage', 'roshan@rstmaintenance.lk', 'rosha')
# print(db.select_user()[0].get_email())
# print(db.select_user()[0].get_password())
# print(db.login('rs', 'll'))

# x = db.select_mark()
#
# for i in x:
#     print(i.get_id())
