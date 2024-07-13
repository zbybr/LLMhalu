def clean_response(response):
    if "YES" in response:
        return "YES"

    elif "NO" in response:
        return "NO"

    elif "NOT SURE" in response:
        return "NOT SURE"

    return response
