def clean_response(response):
    response = response.strip()

    if response.startswith("NOT SURE"):
        return "NOT SURE"

    elif response.startswith("NO"):
        return "NO"

    elif response.startswith("YES"):
        return "YES"

    return response
