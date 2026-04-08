class EmailGrader:

    def grade(self, observation, action):

        text = observation["email"].lower()
        correct = self._get_label(text)

        score = 0.0

        # CATEGORY (partial credit)
        if action["category"] == correct["category"]:
            score += 0.4
        elif correct["category"] in action["category"]:
            score += 0.2

        # PRIORITY
        if action["priority"] == correct["priority"]:
            score += 0.3

        # RESPONSE QUALITY
        if self._good_response(action["response"], text):
            score += 0.3

        return round(min(score, 1.0), 2)

    def _get_label(self, text):

        is_billing = any(k in text for k in ["refund", "charged", "payment"])
        is_technical = any(k in text for k in ["crash", "login", "error"])

        if is_billing:
            return {"category": "billing", "priority": "high"}

        if is_technical:
            return {"category": "technical", "priority": "high"}

        return {"category": "general", "priority": "low"}

    def _good_response(self, response, text):

        response = response.lower()

        if len(response) < 15:
            return False

        keywords = []

        if "refund" in text or "charged" in text:
            keywords = ["refund", "payment"]

        elif "crash" in text or "error" in text:
            keywords = ["fix", "issue"]

        elif "login" in text:
            keywords = ["login", "account"]

        return any(word in response for word in keywords)