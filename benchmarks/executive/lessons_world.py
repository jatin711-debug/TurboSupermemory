"""A task world for testing whether an assistant learns house rules from outcomes.

Marrowgate Kitchen Supply is a fictional wholesaler. Its staff ask an assistant
short questions. Most task families hide a house rule that a capable model,
given only ``WORLD_BRIEF``, gets wrong by default and gets right once the rule
is known. After each task the assistant is told whether it was right (checked
here, deterministically) and, when it was wrong, gets a one-line correction.

Family kinds:

* ``rule``: a hidden convention; the default answer (``naive``) fails.
* ``control``: same topic and wording as a ``sibling`` rule family, but the
  ordinary answer is right; carrying the sibling's rule over makes it wrong
  (``overapplied``).
* ``drift``: a rule that changes at the stream's midpoint (phase 0, phase 1).

The model's answer is the text after its last ``Answer:`` line; ``check``
receives that text.

    python lessons_world.py --selftest
    python lessons_world.py --show refund_approver [--seed 3]

Standard library only.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from typing import Callable

WORLD_BRIEF = (
    "Marrowgate Kitchen Supply is a mid-sized wholesaler of commercial kitchen "
    "equipment and consumables, selling to restaurants, hotels, schools and care "
    "homes from one warehouse and a small sales office. It also services and "
    "repairs the equipment it sells. Its sales, warehouse and accounts staff use "
    "you, an assistant, for quick questions; prices are in US dollars."
)

# --------------------------------------------------------------------------
# Slot vocabularies

CUSTOMERS = [
    "Harbor Lane Bistro", "St. Aldric's School", "the Copperleaf Hotel",
    "Mill Street Diner", "Juniper Hall Cafe", "Riverside Care Home",
    "Northgate Grammar", "The Saltbox Kitchen", "the Larkspur Inn",
    "Brookfield Hospital canteen", "Fennick's Steakhouse", "Oriel Court Hotel",
]
SUPPLIERS = [
    "Kessling Steelworks", "Brannock Appliances", "Tidewell Packaging",
    "Orrin & Daye Cleaning", "Vantrell Refrigeration", "Pellham Glassware",
]
PRODUCTS = [
    "convection oven", "dough mixer", "meat slicer", "dishwasher",
    "deep fryer", "blast chiller", "commercial ice machine", "prep fridge",
    "countertop induction hob", "bain-marie", "salamander grill", "food processor",
]
SPECIALS = [
    "combi oven", "pizza deck oven", "tilting braising pan",
    "pass-through dishwasher", "rotisserie oven", "cold room panel kit",
]
CONSUMABLES = [
    "bottles of degreaser", "rolls of cling film", "boxes of nitrile gloves",
    "packs of napkins", "tubs of sanitiser tablets", "cans of oven cleaner",
    "packs of piping bags", "rolls of foil",
]
BAYS = ["Bay A", "Bay C", "the mezzanine", "the cold store", "the trade counter",
        "dispatch", "Bay F", "overflow racking"]
TOWNS = ["Ashby Cross", "Kelderton", "Wexmoor", "Pinnock Bay", "Hallow Green",
         "Dunmere", "Calloway", "Sefton Rise"]
REASONS = [
    "a damaged delivery", "a late delivery", "the wrong model being shipped",
    "a double charge on their card", "a faulty thermostat", "a cancelled order",
    "missing parts", "a dented door panel",
]
MONTHS = list(calendar.month_name)[1:]
MON3 = [m[:3].upper() for m in MONTHS]
WEEKDAYS = list(calendar.day_name)


def money(x: float) -> str:
    return f"${x:,.2f}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rand_date(rng: random.Random, start=dt.date(2026, 1, 5), days=540) -> dt.date:
    return start + dt.timedelta(days=rng.randrange(days))


def say_date(d: dt.date, rng: random.Random, weekday: bool = False) -> str:
    style = rng.randrange(3)
    if style == 0:
        s = f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    elif style == 1:
        s = f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    else:
        s = f"the {ordinal(d.day)} of {MONTHS[d.month - 1]} {d.year}"
    if weekday:
        s = f"{WEEKDAYS[d.weekday()]} {s[4:] if s.startswith('the ') else s}"
    return s


def long_date(d: dt.date) -> str:
    return f"{WEEKDAYS[d.weekday()]} {d.day} {MONTHS[d.month - 1]} {d.year}"


def next_business_day(d: dt.date) -> dt.date:
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def h12(hour: int) -> tuple[int, str]:
    return (hour % 12 or 12), ("am" if hour < 12 else "pm")


def say_time12(hour: int, minute: int, rng: random.Random) -> str:
    h, ap = h12(hour)
    words = {15: "quarter past", 30: "half past", 45: "quarter to"}
    style = rng.randrange(3)
    if style == 0 and minute in words and 8 <= hour <= 19:
        part = "in the morning" if hour < 12 else ("in the afternoon" if hour < 18 else "in the evening")
        shown = h if minute != 45 else (hour + 1) % 12 or 12
        return f"{words[minute]} {shown} {part}"
    if style == 1:
        return f"{h}.{minute:02d}{ap}"
    return f"{h}:{minute:02d} {ap}"


def items_phrase(lines: list[tuple[int, str, float]]) -> str:
    parts = [f"{q} {p}{'s' if q != 1 else ''} at {money(u)} each" for q, p, u in lines]
    return " and ".join(parts)


def rand_lines(rng: random.Random) -> list[tuple[int, str, float]]:
    out = []
    for prod in rng.sample(PRODUCTS, rng.choice([1, 1, 2])):
        out.append((rng.randint(1, 12), prod, rng.randrange(1800, 240000, 25) / 100))
    return out


# --------------------------------------------------------------------------
# Check specs (JSON-serializable)


def spec_regex(*patterns: str) -> dict:
    return {"type": "regex", "patterns": list(patterns)}


def spec_number(value: float, tol: float = 0.005, units: str = "") -> dict:
    return {"type": "number", "value": value, "tol": tol, "units": units}


def spec_name(accept: list[str], reject: list[str]) -> dict:
    return {"type": "name", "accept": accept, "reject": reject}


def spec_code(value: str) -> dict:
    return {"type": "code", "value": value}


def spec_date(value: dt.date, anchor: dt.date | None = None) -> dict:
    return {"type": "date", "value": value.isoformat(),
            "anchor": anchor.isoformat() if anchor else None}


# --------------------------------------------------------------------------
# Families


@dataclass(frozen=True)
class Family:
    name: str
    split: str  # dev | test
    kind: str  # rule | control | drift
    rule: str | tuple[str, str]  # tuple (phase 0, phase 1) for drift
    templates: tuple[str, ...]
    make: Callable[[random.Random, int], tuple[dict, dict]]
    sibling: str = ""

    def rule_text(self, phase: int = 0) -> str:
        if isinstance(self.rule, tuple):
            return self.rule[phase]
        return self.rule

    def instance(self, seed, k: int, phase: int = 0) -> dict:
        order = list(range(len(self.templates)))
        random.Random(f"{seed}|{self.name}|templates").shuffle(order)
        template = self.templates[order[k % len(order)]]
        rng = random.Random(f"{seed}|{self.name}|{k}")
        slots, out = self.make(rng, phase)
        prompt = template.format(**slots)
        prompt = prompt[0].upper() + prompt[1:]
        task = {
            "id": f"{self.name}-{k:02d}",
            "family": self.name,
            "split": self.split,
            "kind": self.kind,
            "prompt": prompt,
            "check": out["check"],
            "reference": out["reference"],
            "naive": out["naive"],
            "correction": out["correction"],
            "phase": phase,
        }
        if "overapplied" in out:
            task["overapplied"] = out["overapplied"]
        return task


# ---- dev rule: invoice date format -----------------------------------------

def _invoice_date(rng, phase):
    d = rand_date(rng)
    mon = MON3[d.month - 1]
    ref = f"{d.day:02d}-{mon}-{d.year}"
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS),
                 qty=rng.randint(2, 9), said=say_date(d, rng))
    return slots, {
        "check": spec_regex(rf"(?<![0-9])0?{d.day}\s*-\s*{mon}\s*-\s*{d.year}(?![0-9])"),
        "reference": ref,
        "naive": f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
        "correction": f"No - on our invoices we write the date as day, three-letter month in capitals, year: {ref}.",
    }


INVOICE_DATE = Family(
    "invoice_date", "dev", "rule",
    "On Marrowgate's own customer invoices, dates are written DD-MON-YYYY with the month as "
    "three capital letters (e.g. 05-MAR-2026).",
    (
        "What date do I put on the invoice for {cust}? It's dated {said}.",
        "I'm finishing the invoice for the {prod} we sold {cust}; the invoice date is {said}. How should I write the date?",
        "Write the invoice date for an invoice we're issuing on {said}.",
        "{cust} needs a corrected invoice with the issue date {said}. What goes in the date field?",
        "Quick one: how do I write {said} as the date on one of our invoices?",
        "Our invoice to {cust} for {qty} {prod}s goes out {said}. What's the date line on it?",
        "How do I enter the invoice date on a customer invoice issued {said}?",
        "Typing up the invoice for {cust}, date of issue {said} - what exactly do I write in the date box?",
    ),
    _invoice_date,
)


# ---- dev control: a form that states its own date format --------------------

_DATE_FORMATS = {
    "YYYY-MM-DD": (lambda d: f"{d.year}-{d.month:02d}-{d.day:02d}",
                   lambda d: rf"(?<![0-9]){d.year}-0?{d.month}-0?{d.day}(?![0-9])"),
    "MM/DD/YYYY": (lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}",
                   lambda d: rf"(?<![0-9])0?{d.month}/0?{d.day}/{d.year}(?![0-9])"),
    "DD/MM/YYYY": (lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
                   lambda d: rf"(?<![0-9])0?{d.day}/0?{d.month}/{d.year}(?![0-9])"),
}


def _portal_date(rng, phase):
    d = rand_date(rng)
    fmt = rng.choice(list(_DATE_FORMATS))
    render, pattern = _DATE_FORMATS[fmt]
    ref = render(d)
    slots = dict(supplier=rng.choice(SUPPLIERS), fmt=fmt, said=say_date(d, rng))
    return slots, {
        "check": spec_regex(pattern(d)),
        "reference": ref,
        "naive": ref,
        "overapplied": f"{d.day:02d}-{MON3[d.month - 1]}-{d.year}",
        "correction": f"No - that form asks for {fmt}, so it's {ref}. Our DD-MON-YYYY style is only for our own invoices.",
    }


PORTAL_DATE = Family(
    "portal_date", "dev", "control", "",
    (
        "The {supplier} ordering portal wants the delivery date as {fmt}. What do I type for {said}?",
        "The customs declaration asks for the shipment date in {fmt} format. The date is {said} - what do I enter?",
        "The bank's transfer form says the value date must be {fmt}. Payment date is {said}; what goes in the box?",
        "The courier's booking site needs the pickup date as {fmt}. Pickup is {said} - what do I put?",
        "For the {supplier} warranty registration, the purchase date has to be entered as {fmt}. We bought it on {said}.",
        "The council's food-safety form wants the inspection date written {fmt}. The inspection was {said}.",
        "How do I enter {said} in the {supplier} supplier portal? The field is labelled {fmt}.",
        "The payroll spreadsheet's date column is formatted {fmt}. Enter {said} for me.",
    ),
    _portal_date,
    sibling="invoice_date",
)


# ---- dev rule: shipping weight in kg, rounded up ------------------------------

def _bol_weight(rng, phase):
    while True:
        lb = rng.choice([rng.randint(18, 480), rng.randint(180, 4800) / 10])
        kg = lb * 0.45359237
        if abs(kg - round(kg)) > 0.05:
            break
    lb_s = f"{lb:g}"
    ref = f"{math.ceil(kg)} kg"
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS),
                 qty=rng.randint(2, 6), lb=lb_s)
    return slots, {
        "check": spec_number(math.ceil(kg), tol=0, units=r"kg|kilo"),
        "reference": ref,
        "naive": f"{lb_s} lb",
        "correction": f"No - shipping paperwork is always in kilograms, rounded up to the next whole kilo, so {ref}.",
    }


BOL_WEIGHT = Family(
    "bol_weight", "dev", "rule",
    "Weights on bills of lading and other shipping paperwork are stated in kilograms, rounded up "
    "to the next whole kilogram (1 lb = 0.4536 kg).",
    (
        "The {prod} for {cust} weighs {lb} lb. What weight goes on the bill of lading?",
        "Shipping a crate to {cust}: {lb} pounds. What do I write as the weight on the paperwork?",
        "What's the declared weight for the {prod} shipment? The scale says {lb} lb.",
        "Bill of lading for {cust} - the pallet is {lb} lbs. What goes in the weight field?",
        "Our scale read {lb} lb for the {prod}. What shipping weight do I enter?",
        "I need the weight for the freight docs: {qty} {prod}s, {lb} lb in total.",
        "How should I state the weight of a {lb}-pound {prod} on the shipping label?",
        "Prepping the consignment note for {cust}. Gross weight is {lb} lb - what do I put down?",
    ),
    _bol_weight,
)


# ---- dev rule: refund approver by amount -------------------------------------

def _refund_approver(rng, phase):
    amount = rng.choice([rng.randrange(4000, 49999) / 100, rng.randrange(50000, 400000) / 100, 500.0])
    if amount < 500:
        who, accept, reject = "Priya Nair", ["Priya", "Nair"], ["Tomasz", "Weller"]
    else:
        who, accept, reject = "Tomasz Weller", ["Tomasz", "Weller"], ["Priya", "Nair"]
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS),
                 amount=money(amount), reason=rng.choice(REASONS))
    return slots, {
        "check": spec_name(accept, reject),
        "reference": who,
        "naive": "The customer service manager",
        "correction": f"No - refunds under $500 go to Priya Nair; $500 and over need Tomasz Weller in finance. This one goes to {who}.",
    }


REFUND_APPROVER = Family(
    "refund_approver", "dev", "rule",
    "Customer refunds under $500 are approved by Priya Nair (customer service lead); refunds of "
    "$500 or more are approved by Tomasz Weller (finance).",
    (
        "{cust} wants a refund of {amount} for {reason}. Who approves it?",
        "Who needs to sign off a {amount} refund to {cust}?",
        "I've got a refund request from {cust}: {amount}, because of {reason}. Who do I send it to for approval?",
        "Refund of {amount} on a {prod} for {cust} - whose approval do I need?",
        "Who can approve refunding {cust} {amount}? It's for {reason}.",
        "A customer is owed {amount} back because of {reason}. Who approves the refund?",
        "Before I process {cust}'s {amount} refund, who has to approve it?",
        "Which person approves a refund of {amount} for {reason}?",
    ),
    _refund_approver,
)


# ---- dev rule: handling charge on quotes -------------------------------------

def _goods(rng):
    lines = rand_lines(rng)
    goods = round(sum(q * u for q, _, u in lines), 2)
    return lines, goods


def _quote_handling(rng, phase):
    lines, goods = _goods(rng)
    total = round(goods + 35, 2)
    slots = dict(cust=rng.choice(CUSTOMERS), items=items_phrase(lines))
    return slots, {
        "check": spec_number(total, tol=0.005),
        "reference": money(total),
        "naive": money(goods),
        "correction": f"No - every customer quote gets our flat $35 handling charge on top, so it's {money(total)}.",
    }


QUOTE_HANDLING = Family(
    "quote_handling", "dev", "rule",
    "Every customer quote includes a flat $35 handling charge, added once per quote on top of the goods total.",
    (
        "What's the total on a quote for {cust}: {items}?",
        "{cust} wants a quote for {items}. What total do I quote?",
        "Can you work out the quote total for {items}?",
        "Quoting {cust} for {items} - what's the final figure?",
        "How much should the quote to {cust} come to? It's {items}.",
        "Prepare the total for a customer quote: {items}.",
        "I'm sending {cust} a quote for {items}. What's the bottom line?",
        "What do we quote {cust} in total for {items}?",
    ),
    _quote_handling,
)


# ---- dev control: totals that are not customer quotes ------------------------

def _purchase_total(rng, phase):
    lines, goods = _goods(rng)
    slots = dict(supplier=rng.choice(SUPPLIERS), items=items_phrase(lines))
    return slots, {
        "check": spec_number(goods, tol=0.005),
        "reference": money(goods),
        "naive": money(goods),
        "overapplied": money(goods + 35),
        "correction": f"No - it's just {money(goods)}; the $35 handling charge only goes on our quotes to customers.",
    }


PURCHASE_TOTAL = Family(
    "purchase_total", "dev", "control", "",
    (
        "{supplier} invoiced us for {items}. How much do we owe them?",
        "What's the total on our purchase order to {supplier} for {items}?",
        "We've got {items} sitting in stock. What's that stock worth at cost?",
        "How much is the {supplier} bill for {items}?",
        "I'm paying {supplier} for {items}. What amount do I transfer?",
        "Our purchase order to {supplier} is {items}. What's the order total?",
        "The stock count found {items}. What's the total cost value?",
        "{supplier} delivered {items} and sent the bill. What's the total we pay?",
    ),
    _purchase_total,
    sibling="quote_handling",
)


# ---- dev rule: order reference format ----------------------------------------

def _order_ref(rng, phase):
    d = rand_date(rng)
    num = rng.choice([rng.randint(12, 999), rng.randint(1000, 99999)])
    ref = f"MG{d.year % 100:02d}-{num:06d}"
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS), num=num, said=say_date(d, rng))
    return slots, {
        "check": spec_code(ref),
        "reference": ref,
        "naive": f"#{num}",
        "correction": f"No - order references are MG, the two-digit year, a dash and the order number padded to six digits: {ref}.",
    }


ORDER_REF = Family(
    "order_ref", "dev", "rule",
    "Order references are written MG + the last two digits of the order year + '-' + the order number "
    "padded to six digits (e.g. order 4821 placed in 2026 is MG26-004821).",
    (
        "{cust} placed order number {num} on {said}. How do I write the order reference?",
        "What's the order reference for order {num}, placed {said}?",
        "I need to give {cust} their order reference - it's order #{num} from {said}.",
        "Order {num} (a {prod}, ordered {said}) - what's its reference in our format?",
        "How should I write the reference for {cust}'s order {num}? They ordered on {said}.",
        "Put the order reference on the packing slip: order no. {num}, dated {said}.",
        "{cust} is asking for their order reference. It's order {num}, placed {said}.",
        "What reference do I put in the email subject for order {num} from {said}?",
    ),
    _order_ref,
)


# ---- dev rule: same-day shipping cut-off -------------------------------------

def _ship_cutoff(rng, phase):
    while True:
        d = rand_date(rng)
        if d.weekday() < 5:
            break
    minute_of_day = rng.randrange(11 * 60, 14 * 60, 5)
    hour, minute = divmod(minute_of_day, 60)
    if rng.random() < 0.5:
        t = f"{hour}:{minute:02d}"
    else:
        h, ap = h12(hour)
        t = f"{h}:{minute:02d} {ap}"
    day = say_date(d, rng, weekday=True)
    when = rng.choice([f"{day} at {t}", f"{t} on {day}"])
    ship = next_business_day(d)
    ref = long_date(ship)
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS),
                 qty=rng.randint(2, 9), when=when)
    return slots, {
        "check": spec_date(ship, anchor=d),
        "reference": ref,
        "naive": f"{long_date(d)} (same day)",
        "correction": f"No - our cut-off is 11:00; orders at or after that ship the next business day, so {ref}.",
    }


SHIP_CUTOFF = Family(
    "ship_cutoff", "dev", "rule",
    "Orders placed before 11:00 ship the same business day; orders placed at 11:00 or later ship the "
    "next business day (nothing ships at weekends).",
    (
        "{cust} placed an order on {when}. When does it ship?",
        "An order came in {when}. What day will it go out?",
        "If {cust}'s order for {qty} {prod}s was placed {when}, what's the ship date?",
        "Order received {when} - when do we dispatch it?",
        "What ship date do I tell {cust}? They ordered {when}.",
        "When will a {prod} ordered {when} leave the warehouse?",
        "{cust} ordered {when}. When does the order ship?",
        "A customer order was logged {when}. Which day does it ship?",
    ),
    _ship_cutoff,
)


# ---- dev drift: who signs purchase orders ------------------------------------

def _po_signoff(rng, phase):
    if phase == 0:
        who, accept, reject = "Dana Okoro", ["Dana", "Okoro"], ["Felix", "Amsel"]
        corr = "No - purchase orders are signed off by Dana Okoro."
    else:
        who, accept, reject = "Felix Amsel", ["Felix", "Amsel"], ["Dana", "Okoro"]
        corr = "No - since the reorganisation Felix Amsel signs off purchase orders, not Dana Okoro."
    slots = dict(supplier=rng.choice(SUPPLIERS), qty=rng.randint(2, 40), prod=rng.choice(PRODUCTS),
                 amount=money(rng.randrange(30000, 2500000, 5) / 100))
    return slots, {
        "check": spec_name(accept, reject),
        "reference": who,
        "naive": "The purchasing manager",
        "correction": corr,
    }


PO_SIGNOFF = Family(
    "po_signoff", "dev", "drift",
    (
        "Purchase orders to suppliers are signed off by Dana Okoro (purchasing manager).",
        "Purchase orders to suppliers are now signed off by Felix Amsel (operations director); "
        "Dana Okoro no longer signs them.",
    ),
    (
        "Who signs off the purchase order for {qty} {prod}s from {supplier}?",
        "I've drafted a {amount} PO to {supplier}. Who needs to sign it off?",
        "Whose sign-off do I need on a purchase order to {supplier}?",
        "Purchase order for {supplier}, {amount} - who approves it?",
        "Who do I send the {supplier} purchase order to for sign-off?",
        "We're reordering {qty} {prod}s from {supplier}. Who signs the PO?",
        "Who signs purchase orders here? This one's {amount} to {supplier}.",
        "Before the {supplier} PO goes out, who has to sign it off?",
    ),
    _po_signoff,
)


# ---- test rule: time format on delivery notes --------------------------------

def _random_time(rng) -> tuple[int, int]:
    return rng.randint(7, 18), rng.choice(range(0, 60, 5))


def _delivery_note_time(rng, phase):
    hour, minute = _random_time(rng)
    h, ap = h12(hour)
    ref = f"{hour:02d}h{minute:02d}"
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS), said=say_time12(hour, minute, rng))
    return slots, {
        "check": spec_regex(rf"(?<![0-9:.])0?{hour}\s*h\s*{minute:02d}(?![0-9])"),
        "reference": ref,
        "naive": f"{h}:{minute:02d} {ap.upper()}",
        "correction": f"No - on delivery notes and work orders we write 24-hour time with an h, like {ref}.",
    }


DELIVERY_NOTE_TIME = Family(
    "delivery_note_time", "test", "rule",
    "Times on delivery notes and work orders are written in 24-hour form with 'h' in place of the colon "
    "(e.g. 14h30, 09h05).",
    (
        "What time do I write on the delivery note for the drop at {cust}? It's {said}.",
        "The {prod} for {cust} is due at {said}. How do I write that time on the delivery note?",
        "Fill in the arrival time on today's work order: the engineer got there at {said}.",
        "How should {said} be written on a delivery note?",
        "The delivery to {cust} was signed for at {said}. What goes in the time box on the note?",
        "The service visit at {cust} is booked for {said}. What time do I put on the work order?",
        "I'm writing up the delivery note - the van left {cust} at {said}. What goes in the time field?",
        "Write the delivery time {said} the way it goes on our delivery notes.",
    ),
    _delivery_note_time,
)


# ---- test control: a form that states its own time format ---------------------

def _booking_time(rng, phase):
    hour, minute = _random_time(rng)
    h, ap = h12(hour)
    if rng.random() < 0.5:
        fmt = "12-hour (hh:mm AM/PM)"
        said = f"{hour:02d}:{minute:02d}"
        ref = f"{h}:{minute:02d} {ap.upper()}"
        pattern = rf"(?<![0-9:.])0?{h}:{minute:02d}\s*{ap[0]}\.?\s*m(?![a-z])"
    else:
        fmt = "24-hour (HH:MM)"
        said = say_time12(hour, minute, rng)
        ref = f"{hour:02d}:{minute:02d}"
        pattern = rf"(?<![0-9:.])0?{hour}:{minute:02d}(?![0-9])(?!\s*[ap]\.?\s*m(?![a-z]))"
    slots = dict(fmt=fmt, said=said, supplier=rng.choice(SUPPLIERS), cust=rng.choice(CUSTOMERS))
    return slots, {
        "check": spec_regex(pattern),
        "reference": ref,
        "naive": ref,
        "overapplied": f"{hour:02d}h{minute:02d}",
        "correction": f"No - that form wants {fmt}, so it's {ref}. The 'h' style is only for our own delivery notes.",
    }


BOOKING_TIME = Family(
    "booking_time", "test", "control", "",
    (
        "The courier's booking site takes the collection time in {fmt} format. Collection is at {said} - what do I enter?",
        "{supplier}'s portal asks for the delivery window start as {fmt}. It's {said}.",
        "{cust}'s visitor form wants our arrival time in {fmt}. We arrive at {said}; what do I type?",
        "The bank's appointment form takes times as {fmt}. The meeting's at {said}.",
        "The freight company's web form needs the pickup time in {fmt}. Pickup is {said} - what goes in?",
        "How do I enter {said} in the {supplier} booking system? The field says {fmt}.",
        "The calendar tool wants the start time as {fmt}. The training starts at {said}.",
        "The trade show venue's form asks for our setup time in {fmt}. We set up at {said}.",
    ),
    _booking_time,
    sibling="delivery_note_time",
)


# ---- test rule: transfer slips in cases of 24 ----------------------------------

def _units(rng) -> tuple[int, str]:
    while True:
        n = rng.randint(25, 400)
        if n % 24:
            return n, rng.choice(CONSUMABLES)


def _transfer_cases(rng, phase):
    n, items = _units(rng)
    cases = math.ceil(n / 24)
    bay_a, bay_b = rng.sample(BAYS, 2)
    slots = dict(n=n, items=items, a=bay_a, b=bay_b)
    return slots, {
        "check": spec_number(cases, tol=0),
        "reference": f"{cases} cases",
        "naive": f"{n}",
        "correction": f"No - transfer slips count in cases of 24, rounded up, so {n} {items} is {cases} cases.",
    }


TRANSFER_CASES = Family(
    "transfer_cases", "test", "rule",
    "Internal stock transfer slips count quantities in cases of 24 units, rounded up to whole cases.",
    (
        "Moving {n} {items} from {a} to {b}. What quantity do I write on the transfer slip?",
        "I'm transferring {n} {items} to {b}. What goes in the quantity box on the slip?",
        "Transfer slip for {n} {items}, {a} to {b} - quantity?",
        "How do I fill in the quantity on a stock transfer of {n} {items}?",
        "{b} needs {n} {items} from {a}. What quantity goes on the transfer slip?",
        "What's the transfer-slip quantity for shifting {n} {items} out of {a}?",
        "We're restocking {b} with {n} {items}. Quantity for the transfer slip?",
        "Write up the quantity for an internal transfer of {n} {items} from {a}.",
    ),
    _transfer_cases,
)


# ---- test control: customer-facing quantities ---------------------------------

def _invoice_qty(rng, phase):
    n, items = _units(rng)
    slots = dict(n=n, items=items, cust=rng.choice(CUSTOMERS))
    return slots, {
        "check": spec_number(n, tol=0),
        "reference": f"{n}",
        "naive": f"{n}",
        "overapplied": f"{math.ceil(n / 24)} cases",
        "correction": f"No - customers are invoiced per unit, so it's {n}. Cases of 24 are only for our internal transfer slips.",
    }


INVOICE_QTY = Family(
    "invoice_qty", "test", "control", "",
    (
        "{cust} ordered {n} {items}. What quantity goes on their invoice line?",
        "I'm writing the order confirmation for {cust}: {n} {items}. What quantity do I show?",
        "Quantity for the quote line - {cust} wants {n} {items}?",
        "What quantity goes on the delivery note to {cust} for {n} {items}?",
        "{cust} is returning {n} {items}. What quantity goes on the credit note?",
        "How do I fill in the quantity on {cust}'s invoice? They bought {n} {items}.",
        "The receipt for a counter sale of {n} {items} - what quantity do I print?",
        "Customer order line: {n} {items} for {cust}. Quantity field?",
    ),
    _invoice_qty,
    sibling="transfer_cases",
)


# ---- test rule: who books a delivery, by distance -----------------------------

def _delivery_booking(rng, phase):
    km = rng.choice([rng.randint(3, 39), rng.randint(41, 180), 40])
    if km > 40:
        who, accept, reject = "Halden Freight", ["Halden"], ["Marta", "Ruiz"]
    else:
        who, accept, reject = "Marta Ruiz", ["Marta", "Ruiz"], ["Halden"]
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS), qty=rng.randint(2, 6),
                 km=km, town=rng.choice(TOWNS))
    return slots, {
        "check": spec_name(accept, reject),
        "reference": who,
        "naive": "The dispatch team",
        "correction": f"No - over 40 km we book Halden Freight; 40 km or less goes on our own van, booked with Marta Ruiz. This one is {who}.",
    }


DELIVERY_BOOKING = Family(
    "delivery_booking", "test", "rule",
    "Deliveries more than 40 km from the warehouse are booked through Halden Freight; deliveries of "
    "40 km or less go on Marrowgate's own van, booked with dispatcher Marta Ruiz.",
    (
        "Who books the delivery of a {prod} to {cust}? They're {km} km from the warehouse.",
        "{cust} is {km} km away and wants their {prod} delivered next week. Who do I book that with?",
        "We need to get {qty} {prod}s out to {cust}, {km} km from us. Who arranges the delivery?",
        "Delivery to {town}, {km} km from the warehouse - who should book it?",
        "Who handles booking a drop at {cust}? Distance from the warehouse is {km} km.",
        "I've got a {prod} going to a site {km} km out. Who do I ask to book the delivery?",
        "{cust} ({km} km away) just paid for their {prod}. Who books the delivery?",
        "Who do I contact to schedule delivery to {cust} in {town}, {km} km from the warehouse?",
    ),
    _delivery_booking,
)


# ---- test rule: labour billed in started 15-minute blocks ----------------------

def _labour_billing(rng, phase):
    while True:
        minutes = rng.randint(20, 200)
        if minutes % 15:
            break
    rate = rng.choice([60, 72, 80, 84, 90, 96, 100, 120])
    billed = round(math.ceil(minutes / 15) * rate / 4, 2)
    naive = round(minutes * rate / 60, 2)
    if minutes >= 60 and rng.random() < 0.6:
        hrs, mins = divmod(minutes, 60)
        dur = rng.choice([f"{hrs} hour{'s' if hrs > 1 else ''} {mins} minutes", f"{hrs} h {mins} min"])
    else:
        dur = f"{minutes} minutes"
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(PRODUCTS), dur=dur, rate=rate)
    return slots, {
        "check": spec_number(billed, tol=0.005),
        "reference": money(billed),
        "naive": money(naive),
        "correction": f"No - labour is billed in 15-minute blocks and any started block counts in full, so it's {money(billed)}.",
    }


LABOUR_BILLING = Family(
    "labour_billing", "test", "rule",
    "Repair labour is billed in 15-minute blocks at the hourly rate; any started block is charged in full.",
    (
        "A technician spent {dur} repairing {cust}'s {prod}. Labour is ${rate}/hour - what's the labour charge?",
        "What do we bill for {dur} of repair labour at ${rate} an hour?",
        "{cust}'s {prod} repair took {dur}. At ${rate}/hr, how much labour do I put on the invoice?",
        "What's the labour charge for a {dur} service call at ${rate} per hour?",
        "Our tech was on the {prod} for {dur}. What's the labour line at ${rate}/h?",
        "How much labour do we charge {cust} for {dur} of work at ${rate} an hour?",
        "Repair job: {dur} on a {prod}, labour rate ${rate}/hour. Labour total?",
        "Work out the labour for {cust}: {dur} at ${rate} per hour.",
    ),
    _labour_billing,
)


# ---- test rule: return authorisation numbers ----------------------------------

def _rma_number(rng, phase):
    d = rand_date(rng)
    acct = rng.randint(1000, 99999)
    ref = f"RA{acct}-{d.year % 100:02d}{d.month:02d}{d.day:02d}"
    slots = dict(cust=rng.choice(CUSTOMERS), acct=acct, prod=rng.choice(PRODUCTS),
                 qty=rng.randint(2, 6), said=say_date(d, rng))
    return slots, {
        "check": spec_code(ref),
        "reference": ref,
        "naive": f"RMA-{acct}",
        "correction": f"No - return numbers are RA, the account number, a dash, then the return date as YYMMDD: {ref}.",
    }


RMA_NUMBER = Family(
    "rma_number", "test", "rule",
    "Return authorisation numbers are RA + the customer's account number + '-' + the return date as "
    "YYMMDD (e.g. account 7731 returning on 4 Nov 2026 gets RA7731-261104).",
    (
        "{cust} (account {acct}) is returning a {prod} on {said}. What return authorisation number do I give them?",
        "Issue a return number for account {acct}; the return is dated {said}.",
        "What's the return authorisation number for a return from account {acct} on {said}?",
        "Customer account {acct} is sending back {qty} {prod}s on {said}. What's the return number?",
        "{cust}, account no. {acct}, wants to return their {prod}. Return date {said} - what return authorisation do I issue?",
        "I need a return authorisation for {cust} (acct {acct}), returning on {said}.",
        "Generate the return number for account {acct}'s return of a {prod} on {said}.",
        "What return authorisation number goes on the label for account {acct}? Return date: {said}.",
    ),
    _rma_number,
)


# ---- test rule: supplier invoices due end of month + 30 ------------------------

def _supplier_due(rng, phase):
    while True:
        d = rand_date(rng)
        if d.day != calendar.monthrange(d.year, d.month)[1]:
            break
    eom = d.replace(day=calendar.monthrange(d.year, d.month)[1])
    due = eom + dt.timedelta(days=30)
    naive = d + dt.timedelta(days=30)
    slots = dict(supplier=rng.choice(SUPPLIERS), said=say_date(d, rng),
                 amount=money(rng.randrange(20000, 1500000, 5) / 100))
    ref = f"{due.day} {MONTHS[due.month - 1]} {due.year}"
    return slots, {
        "check": spec_date(due),
        "reference": ref,
        "naive": f"{naive.day} {MONTHS[naive.month - 1]} {naive.year}",
        "correction": f"No - supplier invoices are due 30 days after the end of the month they're dated in, so {ref}.",
    }


SUPPLIER_DUE = Family(
    "supplier_due", "test", "rule",
    "Supplier invoices are due 30 days after the end of the month in which they are dated.",
    (
        "{supplier}'s invoice is dated {said}. When is it due for payment?",
        "When do we have to pay a {supplier} invoice dated {said}?",
        "Supplier invoice for {amount} from {supplier}, issued {said} - what's the due date?",
        "What payment due date do I enter for a supplier invoice dated {said}?",
        "{supplier} billed us {amount} on {said}. By when must it be paid?",
        "What's the due date for the {supplier} invoice of {said}?",
        "I'm logging a {supplier} invoice dated {said} in accounts payable. When is it due?",
        "We received an invoice from {supplier} dated {said} for {amount}. When do we pay it?",
    ),
    _supplier_due,
)


# ---- test drift: special-order deposit -----------------------------------------

def _special_order_deposit(rng, phase):
    price = rng.randrange(120, 1500) * 10
    pct = 35 if phase == 0 else 15
    deposit = round(price * pct / 100, 2)
    if phase == 0:
        corr = f"No - special orders take a 35% deposit, so {money(deposit)}."
    else:
        corr = f"No - the special-order deposit is 15% now, not 35%, so {money(deposit)}."
    slots = dict(cust=rng.choice(CUSTOMERS), prod=rng.choice(SPECIALS), price=money(price))
    return slots, {
        "check": spec_number(deposit, tol=0.005),
        "reference": money(deposit),
        "naive": money(price * 0.5),
        "correction": corr,
    }


SPECIAL_ORDER_DEPOSIT = Family(
    "special_order_deposit", "test", "drift",
    (
        "Special-order items require a 35% deposit when the order is placed.",
        "Special-order items now require a 15% deposit when the order is placed (it used to be 35%).",
    ),
    (
        "{cust} is special-ordering a {prod} for {price}. How much deposit do we take?",
        "What deposit do I collect on a special order worth {price}?",
        "Special order for {cust}: a {prod} at {price}. Deposit amount?",
        "How much does {cust} pay up front on their {price} special order?",
        "A customer wants a {prod} we don't stock - {price}, special order. What deposit do we ask for?",
        "What's the deposit on a {price} special-order {prod}?",
        "I'm taking a special order from {cust} for {price}. What deposit goes on the receipt?",
        "{cust} wants us to order in a {prod} ({price}). How much deposit do they pay now?",
    ),
    _special_order_deposit,
)


FAMILIES: list[Family] = [
    INVOICE_DATE, BOL_WEIGHT, REFUND_APPROVER, QUOTE_HANDLING, ORDER_REF, SHIP_CUTOFF,
    PORTAL_DATE, PURCHASE_TOTAL, PO_SIGNOFF,
    DELIVERY_NOTE_TIME, TRANSFER_CASES, DELIVERY_BOOKING, LABOUR_BILLING, RMA_NUMBER, SUPPLIER_DUE,
    BOOKING_TIME, INVOICE_QTY, SPECIAL_ORDER_DEPOSIT,
]
_BY_NAME = {f.name: f for f in FAMILIES}


# --------------------------------------------------------------------------
# Public API


def families(split: str = "all") -> list[Family]:
    if split not in ("dev", "test", "all"):
        raise ValueError(f"split must be dev, test or all, not {split!r}")
    return [f for f in FAMILIES if split == "all" or f.split == split]


def family(name: str) -> Family:
    return _BY_NAME[name]


def stream(split: str, seed, per_family: int = 12) -> list[dict]:
    """Instances of the split's families in a seed-dependent order.

    Drift instances take phase 0 in the first half of the stream and phase 1
    in the second half, by position.
    """
    slots = [(f, k) for f in families(split) for k in range(per_family)]
    random.Random(f"{seed}|{split}|order").shuffle(slots)
    half = len(slots) // 2
    out = []
    for pos, (fam, k) in enumerate(slots):
        phase = int(pos >= half) if fam.kind == "drift" else 0
        out.append(fam.instance(seed, k, phase))
    return out


def rule_for(task: dict) -> str:
    return _BY_NAME[task["family"]].rule_text(task.get("phase", 0))


def rules_text(split: str = "all", phase: int = 0) -> str:
    """Every rule of the split, stated plainly (drift rules at ``phase``)."""
    lines = [f"- {f.rule_text(phase)}" for f in families(split) if f.kind != "control"]
    return "House rules at Marrowgate Kitchen Supply:\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# Checking

_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "−": "-"})


def _clean(answer: str) -> str:
    text = (answer or "").translate(_DASHES).strip()
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip("`*\"'").strip()
    return re.sub(r"\s+", " ", text)


_NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")
_TAIL_MARK = re.compile(r"=|→|->|\bcomes to\b|\bequals\b|\bis\b|\btotals\b", re.I)


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        rest = text[m.end():].lstrip()
        if rest.startswith("%") or rest.lower().startswith("percent"):
            continue
        out.append(float(m.group()))
    return out


def _answer_number(text: str) -> float | None:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    marks = list(_TAIL_MARK.finditer(text))
    if marks:
        tail = _numbers(text[marks[-1].end():])
        if tail:
            return tail[0]
    nums = _numbers(text)
    return nums[0] if nums else None


def _first_pos(text: str, tokens: list[str]) -> int | None:
    hits = [m.start() for t in tokens for m in re.finditer(rf"\b{re.escape(t)}\b", text, re.I)]
    return min(hits) if hits else None


_MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_MONTH_IDX = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)}


def _mk(y: int, m: int, d: int) -> set[dt.date]:
    try:
        return {dt.date(y, m, d)}
    except ValueError:
        return set()


def _date_mentions(text: str, default_year: int, anchor: dt.date | None) -> set[dt.date]:
    """The date(s) the first date mention in ``text`` can mean."""
    explicit: list[tuple[int, set[dt.date]]] = []
    for m in re.finditer(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", text):
        explicit.append((m.start(), _mk(int(m[1]), int(m[2]), int(m[3]))))
    for m in re.finditer(r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](\d{4}|\d{2})(?!\d)", text):
        y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        explicit.append((m.start(), _mk(y, int(m[2]), int(m[1])) | _mk(y, int(m[1]), int(m[2]))))
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}\b\.?,?(?:\s+(\d{{4}}))?",
                         text, re.I):
        y = int(m[3]) if m[3] else default_year
        explicit.append((m.start(), _mk(y, _MONTH_IDX[m[2][:3].lower()], int(m[1]))))
    for m in re.finditer(rf"\b{_MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d)(?:,?\s+(\d{{4}}))?", text, re.I):
        y = int(m[3]) if m[3] else default_year
        explicit.append((m.start(), _mk(y, _MONTH_IDX[m[1][:3].lower()], int(m[2]))))
    explicit = [e for e in explicit if e[1]]
    if explicit:
        return min(explicit, key=lambda e: e[0])[1]
    if anchor is None:
        return set()
    relative: list[tuple[int, dt.date]] = []
    for m in re.finditer(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text, re.I):
        wd = WEEKDAYS.index(m[1].capitalize())
        relative.append((m.start(), anchor + dt.timedelta(days=(wd - anchor.weekday()) % 7)))
    for m in re.finditer(r"\bsame (?:business |working )?day\b|\btoday\b", text, re.I):
        relative.append((m.start(), anchor))
    for m in re.finditer(r"\bnext (?:business|working) day\b", text, re.I):
        relative.append((m.start(), next_business_day(anchor)))
    for m in re.finditer(r"\btomorrow\b", text, re.I):
        relative.append((m.start(), anchor + dt.timedelta(days=1)))
    if relative:
        return {min(relative, key=lambda r: r[0])[1]}
    return set()


def check(task: dict, answer: str) -> bool:
    """True when ``answer`` (the text after ``Answer:``) is right for ``task``."""
    spec = task["check"]
    text = _clean(answer)
    kind = spec["type"]
    if kind == "regex":
        return any(re.search(p, text, re.I) for p in spec["patterns"])
    if kind == "number":
        value = _answer_number(text)
        if value is None or abs(value - spec["value"]) > spec["tol"] + 1e-9:
            return False
        return not spec.get("units") or bool(re.search(spec["units"], text, re.I))
    if kind == "name":
        good = _first_pos(text, spec["accept"])
        bad = _first_pos(text, spec["reject"])
        return good is not None and (bad is None or good < bad)
    if kind == "code":
        norm = re.sub(r"\s*-\s*", "-", text.upper())
        return re.search(rf"(?<![A-Z0-9]){re.escape(spec['value'])}(?![A-Z0-9])", norm) is not None
    if kind == "date":
        target = dt.date.fromisoformat(spec["value"])
        anchor = dt.date.fromisoformat(spec["anchor"]) if spec.get("anchor") else None
        return target in _date_mentions(text, target.year, anchor)
    raise ValueError(f"unknown check type {kind!r}")


# --------------------------------------------------------------------------
# Self-test and CLI

# Answers a model might phrase differently from ``reference``; each must pass.
_EXTRA_PASS = {
    "bol_weight": lambda t: [f"{t['reference']} ({t['naive']})", t["reference"].replace(" kg", " kilograms")],
    "quote_handling": lambda t: [f"Total: {t['reference']}", f"goods {t['naive']} + $35 handling = {t['reference']}"],
    "transfer_cases": lambda t: [f"{t['naive']} units = {t['reference']}", t["reference"].split()[0]],
    "invoice_qty": lambda t: [f"{t['reference']} units", f"Quantity: {t['reference']}"],
    "ship_cutoff": lambda t: [t["reference"].split()[0], "Next business day, " + t["reference"]],
    "special_order_deposit": lambda t: [f"{t['reference']} (deposit)"],
    "order_ref": lambda t: [t["reference"].lower(), f"Order reference {t['reference']}"],
}


def _selftest(seeds=(0, 1, 2, 3, 7, 11), per_family: int = 12) -> int:
    failures: list[str] = []
    counts: dict[str, int] = {f.name: 0 for f in FAMILIES}
    phases: dict[str, set[int]] = {f.name: set() for f in FAMILIES if f.kind == "drift"}

    def expect(ok: bool, msg: str):
        if not ok:
            failures.append(msg)

    for fam in FAMILIES:
        expect(len(set(fam.templates)) >= 8, f"{fam.name}: fewer than 8 templates")
        expect(fam.split in ("dev", "test") and fam.kind in ("rule", "control", "drift"), f"{fam.name}: bad split/kind")
        if fam.kind == "control":
            expect(fam.sibling in _BY_NAME and _BY_NAME[fam.sibling].kind == "rule", f"{fam.name}: bad sibling")
        else:
            expect(bool(fam.rule_text(0)), f"{fam.name}: no rule")

    for split in ("dev", "test", "all"):
        for seed in seeds:
            tasks = stream(split, seed, per_family)
            expect(len({t["id"] for t in tasks}) == len(tasks), f"{split}/{seed}: duplicate ids")
            expect(tasks == stream(split, seed, per_family), f"{split}/{seed}: not deterministic")
            half = len(tasks) // 2
            for pos, t in enumerate(tasks):
                fam = _BY_NAME[t["family"]]
                where = f"{split}/seed {seed}/{t['id']} (phase {t['phase']})"
                json.dumps(t)
                if split == "all":
                    counts[fam.name] += 1
                if fam.kind == "drift":
                    phases[fam.name].add(t["phase"])
                    expect(t["phase"] == int(pos >= half), f"{where}: drift phase not by position")
                    other = fam.instance(seed, int(t["id"].rsplit("-", 1)[1]), 1 - t["phase"])
                    expect(not check(other, t["reference"]), f"{where}: reference passes the other phase")
                expect(check(t, t["reference"]), f"{where}: reference fails: {t['reference']!r}")
                expect(check(t, "Answer: " + t["reference"]), f"{where}: 'Answer:' prefix not tolerated")
                naive_ok = check(t, t["naive"])
                if fam.kind == "control":
                    expect(naive_ok, f"{where}: naive should pass: {t['naive']!r}")
                    expect(not check(t, t["overapplied"]), f"{where}: overapplied passes: {t['overapplied']!r}")
                else:
                    expect(not naive_ok, f"{where}: naive should fail: {t['naive']!r}")
                for extra in _EXTRA_PASS.get(fam.name, lambda _t: [])(t):
                    expect(check(t, extra), f"{where}: variant fails: {extra!r}")
                expect(bool(t["correction"]) and "\n" not in t["correction"], f"{where}: bad correction")
                expect("{" not in t["prompt"], f"{where}: unfilled slot in prompt")
            for fam in families(split):
                prompts = [t["prompt"] for t in tasks if t["family"] == fam.name]
                expect(len(set(prompts)) == len(prompts), f"{split}/{seed}/{fam.name}: repeated prompts")
    for name, ph in phases.items():
        expect(ph == {0, 1}, f"{name}: drift phases seen {sorted(ph)}")

    print(f"{'family':24} {'split':5} {'kind':8} {'count':>5}  sibling")
    for fam in FAMILIES:
        print(f"{fam.name:24} {fam.split:5} {fam.kind:8} {counts[fam.name]:>5}  {fam.sibling}")
    print(f"\n{len(seeds)} seeds x {per_family} per family; {sum(counts.values())} instances in the 'all' streams")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:60]:
            print("  " + f)
        return 1
    print("self-test passed")
    return 0


def _show(name: str, seed) -> None:
    fam = _BY_NAME[name]
    phase_list = [0, 1] if fam.kind == "drift" else [0]
    for phase in phase_list:
        for k in range(5):
            t = fam.instance(seed, k, phase)
            print(json.dumps(t, indent=2))
    print(f"\nrule: {fam.rule!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", metavar="FAMILY")
    ap.add_argument("--seed", default="0")
    args = ap.parse_args(argv)
    if args.show:
        if args.show not in _BY_NAME:
            print(f"unknown family {args.show!r}; choose from {', '.join(_BY_NAME)}")
            return 2
        _show(args.show, args.seed)
        return 0
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
