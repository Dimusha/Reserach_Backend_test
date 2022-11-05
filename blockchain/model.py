class User:
    def __init__(self, name, email, password, user_level, image):
        self.name = name
        self.email = email
        self.password = password
        self.user_level = user_level
        self.image = image

    def get_name(self):
        return self.name

    def get_email(self):
        return self.email

    def get_password(self):
        return self.password

    def get_user_level(self):
        return self.user_level

    def get_image(self):
        return self.image

    def get_str(self):
        return '{ "name" : "' + str(self.name) + '", "email" : "' + str(self.email) + '", "password" : "' + str(
            self.password) + '", "user_level" : "' + str(self.user_level) + '", "image" : "' + str(self.image) + '"} '
