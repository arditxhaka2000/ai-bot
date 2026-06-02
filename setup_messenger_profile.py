"""
setup_messenger_profile.py — one-time Messenger UX setup.

Run once (after PAGE_ACCESS_TOKEN is set) to give the chat a polished feel:
  • a greeting shown before the first message
  • a "Get Started" button
  • ice-breaker suggestions (tappable starter questions)
  • a persistent menu (always-visible shortcuts)

    python setup_messenger_profile.py

The payloads it sets (TRACK, PRICING, DELIVERY_TIME, AGENT, …) are plain text
the webhook already understands, so tapping them flows through the normal brain.
"""

import requests

import config

PROFILE_URL = "https://graph.facebook.com/v21.0/me/messenger_profile"

PROFILE = {
    "get_started": {"payload": "GET_STARTED"},
    "greeting": [
        {"locale": "default",
         "text": "Hi! 👋 I'm Cargoteer's assistant. I can track shipments, give "
                 "delivery times & rates, and connect you to a human. Ask me "
                 "anything — in English or Albanian."}
    ],
    "ice_breakers": [
        {"locale": "default",
         "call_to_actions": [
             {"question": "Track my shipment", "payload": "Track my shipment"},
             {"question": "How much does shipping cost?", "payload": "How much does shipping cost?"},
             {"question": "How long does delivery take?", "payload": "How long does delivery take?"},
             {"question": "Talk to a human", "payload": "agent"},
         ]}
    ],
    "persistent_menu": [
        {
            "locale": "default",
            "composer_input_disabled": False,
            "call_to_actions": [
                {"type": "postback", "title": "📦 Track a shipment", "payload": "Track my shipment"},
                {"type": "postback", "title": "💰 Pricing", "payload": "How much does shipping cost?"},
                {"type": "postback", "title": "🧑 Talk to a human", "payload": "agent"},
                {"type": "web_url", "title": "🌐 Website", "url": "https://cargoteer.com"},
            ],
        }
    ],
}


def main():
    if not config.PAGE_ACCESS_TOKEN:
        print("PAGE_ACCESS_TOKEN not set — fill it in .env first.")
        return
    resp = requests.post(
        PROFILE_URL,
        params={"access_token": config.PAGE_ACCESS_TOKEN},
        json=PROFILE,
        timeout=15,
    )
    print(resp.status_code, resp.text)
    if resp.status_code == 200:
        print("✅ Messenger profile configured (greeting, menu, ice breakers).")
    else:
        print("⚠️ Setup failed — check the token and that it has "
              "pages_messaging permission.")


if __name__ == "__main__":
    main()
