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
         "text": "Welcome to Cargoteer! 👋 We dispatch for owner-operators and "
                 "small fleets — best-paying loads, strong rate negotiation, and "
                 "we handle brokers & paperwork. Message us to get started."},
        {"locale": "es_LA",
         "text": "¡Bienvenido a Cargoteer! 👋 Despachamos para operadores-"
                 "propietarios y flotas pequeñas — las cargas mejor pagadas, "
                 "negociación fuerte de tarifas, y manejamos brokers y papeleo. "
                 "Escríbenos para empezar."},
        {"locale": "ru_RU",
         "text": "Добро пожаловать в Cargoteer! 👋 Мы диспетчеры для водителей-"
                 "собственников и небольших автопарков — самые выгодные грузы, "
                 "сильные переговоры по ставкам, берём на себя брокеров и "
                 "документы. Напишите нам, чтобы начать."}
    ],
    "ice_breakers": [
        {"locale": "default",
         "call_to_actions": [
             {"question": "What do you charge?", "payload": "What do you charge?"},
             {"question": "How does dispatch work?", "payload": "How does dispatch work?"},
             {"question": "I want to get started", "payload": "I want to get started"},
             {"question": "Talk to the team", "payload": "talk to the team"},
         ]}
    ],
    "persistent_menu": [
        {
            "locale": "default",
            "composer_input_disabled": False,
            "call_to_actions": [
                {"type": "postback", "title": "💲 Pricing", "payload": "What do you charge?"},
                {"type": "postback", "title": "🚚 How it works", "payload": "How does dispatch work?"},
                {"type": "postback", "title": "📝 Get started", "payload": "I want to get started"},
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
