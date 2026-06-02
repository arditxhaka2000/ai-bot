"""
try_db.py — check which database backend is active and round-trip it.

    python try_db.py

Prints whether you're on Turso (cloud) or local SQLite, then writes and reads
a throwaway record to prove the connection works end to end.
"""

import store


def main():
    print(f"Backend: {store.backend()}")
    if store.backend() == "sqlite":
        import config
        print(f"  (local file: {config.DB_PATH})")
        print("  Set TURSO_DATABASE_URL + TURSO_AUTH_TOKEN to use the cloud DB.")

    store.init()
    sender = "__selftest__"
    store.reset_sender(sender)
    store.add_message(sender, "user", "ping", "test")
    store.add_message(sender, "assistant", "pong", "test")
    hist = store.get_history(sender, 10)
    store.add_lead(sender, "test", "book_shipment", "selftest lead")
    leads = [l for l in store.recent_leads(20) if l["sender"] == sender]
    store.reset_sender(sender)

    ok = (len(hist) == 2 and hist[0]["content"] == "ping" and len(leads) == 1)
    print(f"Round-trip: {'✅ OK' if ok else '❌ FAILED'} "
          f"(history={len(hist)}, leads={len(leads)})")
    if not ok:
        print("Check the [store] errors above (URL/token correct?).")


if __name__ == "__main__":
    main()
