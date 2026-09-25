"""Generate the deterministic mock world: data/orders.json, data/kb.json, data/service_status.json.

Run once; outputs are committed. Order ids are assigned by bucket so tickets can target a known policy branch.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 9, 20)
rng = random.Random(1234)

ITEMS = [
    ("Wireless earbuds", 79.99),
    ("Mechanical keyboard", 129.00),
    ("Running shoes", 95.50),
    ("Coffee grinder", 59.90),
    ("Yoga mat", 34.99),
    ("Smart watch", 249.00),
    ("Backpack", 68.00),
    ("Desk lamp", 42.75),
    ("Bluetooth speaker", 89.00),
    ("Winter jacket", 179.99),
    ("Phone case", 19.99),
    ("Water bottle", 24.50),
    ("Gaming mouse", 59.00),
    ("Air fryer", 119.00),
    ("Office chair", 289.00),
]
NAMES = [
    "alex", "sam", "maria", "omar", "lina", "chen", "priya", "jonas", "fatima", "leo",
    "nadia", "tom", "yuki", "rami", "ines", "kofi", "sara", "ivan", "maya", "diego",
]  # fmt: skip

# bucket -> count. Order matters: ids ORD-10001.. are assigned sequentially.
BUCKETS = [
    ("refundable", 14),  # delivered 2-28 days ago, not refunded, no duplicate
    ("outside_window", 10),  # delivered 35-150 days ago, not refunded, no duplicate
    ("already_refunded", 6),  # delivered, refunded already
    ("duplicate", 8),  # delivered within window, duplicate charge, not refunded
    ("shipped", 8),
    ("processing", 6),
    ("lost", 6),
    ("duplicate_old", 2),  # duplicate charge but delivered outside window (duplicate still wins)
]


def make_order(i: int, bucket: str) -> dict:
    item, amount = ITEMS[(i * 7) % len(ITEMS)]
    name = NAMES[i % len(NAMES)]
    oid = f"ORD-{10001 + i}"
    if bucket in ("refundable", "duplicate", "already_refunded"):
        delivered_ago = rng.randint(2, 28)
    elif bucket in ("outside_window", "duplicate_old"):
        delivered_ago = rng.randint(35, 150)
    else:
        delivered_ago = None
    order_ago = (delivered_ago or 0) + rng.randint(3, 9)
    order_date = TODAY - timedelta(days=order_ago)
    delivered_date = TODAY - timedelta(days=delivered_ago) if delivered_ago is not None else None
    status = {
        "shipped": "shipped",
        "processing": "processing",
        "lost": "lost",
    }.get(bucket, "delivered")
    charges = [{"charge_id": f"CH-{50000 + i * 3}", "amount": amount, "date": order_date.isoformat()}]
    if bucket in ("duplicate", "duplicate_old"):
        charges.append(
            {"charge_id": f"CH-{50000 + i * 3 + 1}", "amount": amount, "date": order_date.isoformat()}
        )
    return {
        "order_id": oid,
        "customer_email": f"{name}{i}@example.com",
        "item": item,
        "amount": amount,
        "currency": "USD",
        "order_date": order_date.isoformat(),
        "delivered_date": delivered_date.isoformat() if delivered_date else None,
        "status": status,
        "refunded": bucket == "already_refunded",
        "charges": charges,
        "_bucket": bucket,  # generator metadata; tools strip it
    }


def main() -> None:
    orders = []
    i = 0
    for bucket, n in BUCKETS:
        for _ in range(n):
            orders.append(make_order(i, bucket))
            i += 1
    (ROOT / "data/orders.json").write_text(json.dumps(orders, indent=1))

    kb = [
        ("KB01", "Reset your password", "password reset forgot login email link",
         "Use 'Forgot password' on the sign-in page. The reset link expires after 30 minutes. Check spam if the email does not arrive."),
        ("KB02", "App crashes on login", "app crash login android ios open close freeze",
         "Update to the latest app version, clear the app cache, and restart the phone. Android v5.2 has a known login crash fixed in v5.2.1."),
        ("KB03", "Shipping delays", "shipping delay late package slow carrier",
         "Carriers may add 2-5 business days during peak periods. Tracking updates can lag by 24 hours."),
        ("KB04", "Two-factor codes not arriving", "2fa two factor code sms authentication verification",
         "Codes can take up to 2 minutes. Make sure the phone number is correct and try the authenticator app option."),
        ("KB05", "Payment declined", "payment declined card failed checkout error",
         "Check the card details and billing address. If the payments service is degraded, retry after 30 minutes."),
        ("KB06", "Change delivery address", "change address delivery shipping update",
         "The address can be changed while an order is processing. Once shipped, contact the carrier."),
        ("KB07", "Cancel an order", "cancel order stop purchase",
         "Orders can be cancelled while they are processing. Shipped orders must be returned after delivery."),
        ("KB08", "Refund timelines", "refund time how long money back bank",
         "Approved refunds reach the original payment method in 5-7 business days."),
        ("KB09", "Track your order", "track tracking where order status package",
         "Use the tracking link in your shipping confirmation email or the Orders page in the app."),
        ("KB10", "App not syncing", "sync data not updating app refresh offline",
         "Pull to refresh, check the internet connection, then sign out and back in to force a sync."),
        ("KB11", "Promo code not working", "promo code coupon discount invalid",
         "Promo codes are case sensitive, cannot be combined, and expire on the date shown in the offer."),
        ("KB12", "Account locked", "account locked too many attempts blocked login",
         "Accounts lock for 15 minutes after 5 failed sign-in attempts. Use password reset to unlock immediately."),
        ("KB13", "Update payment method", "update card payment method billing",
         "Go to Settings > Billing to add or replace a card. The old card is kept until the new one is verified."),
        ("KB14", "Notifications not working", "notifications push alerts not receiving",
         "Enable notifications for the app in the phone settings and make sure battery saver is not blocking them."),
        ("KB15", "Website login errors", "website login error 500 browser sign in page",
         "Clear browser cookies, try a private window, and check the service status page for login incidents."),
    ]  # fmt: skip
    (ROOT / "data/kb.json").write_text(
        json.dumps([{"id": a, "title": t, "keywords": k, "body": b} for a, t, k, b in kb], indent=1)
    )

    status = {
        "payments": {
            "status": "degraded",
            "note": "Delays in card authorizations since 2026-09-19 08:00 UTC.",
        },
        "login": {"status": "degraded", "note": "Elevated login errors on Android app v5.2; fix in v5.2.1."},
        "shipping-api": {"status": "operational", "note": ""},
        "notifications": {"status": "operational", "note": ""},
    }
    (ROOT / "data/service_status.json").write_text(json.dumps(status, indent=1))
    print(f"orders={len(orders)} kb={len(kb)} services={len(status)}")


if __name__ == "__main__":
    main()
