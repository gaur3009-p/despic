class ConversationManager:

    def __init__(self):

        self.users = {}

    def register_user(self, user_id, language):

        self.users[user_id] = {
            "language": language
        }

    def get_target_language(self, user_id):

        for uid in self.users:

            if uid != user_id:

                return self.users[uid]["language"]

        return None
