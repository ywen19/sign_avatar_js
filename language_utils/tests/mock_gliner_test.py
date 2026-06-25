from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

labels = ["person", "city", "region", "country", "restaurant", "gallery", "museum", "address"]

test_cases = [
    # 1. restaurant / venue
    "The Fisherman's Wharf in Bournemouth is a highly recommended seafood restaurant.",
    "The Seafood Restaurant is near Bournemouth beach.",
    "The Fishmonger is popular with locals.",
    "the fishmonger sells fish.",
    "the restaurant serves seafood.",

    # 2. gallery / museum
    "The Bournemouth Art Gallery features works by Turner, Constable, and Monet.",
    "The museum is at 12 Westover Road, Bournemouth.",
    "The National Gallery is in London.",
    "The British Museum is in London.",
    "the museum is open today.",
    "the gallery has paintings.",

    # 3. people
    "John Smith visited Bournemouth.",
    "William Shakespeare wrote plays.",
    "Turner painted landscapes.",
    "Monet is well known.",
    "Constable was an English painter.",

    # 4. city / region / country
    "Bournemouth is in Dorset, England.",
    "Poole is near Bournemouth.",
    "London is larger than Bournemouth.",
    "Paris is in France.",
    "Cardiff is in Wales.",
    "Edinburgh is in Scotland.",

    # 5. address
    "The Fisherman's Wharf is located at 1-3 The Fisherman's Wharf, Bournemouth, Dorset, BH1 1JQ.",
    "The museum is on Westover Road, Bournemouth.",
    "The restaurant is at 12 Westover Road, Bournemouth.",
    "The office is at Flat 2, 10 Sea Road, Bournemouth.",
    "Meet me at Apartment 4B, 22 Hill Street, Poole.",

    # 6. postcode / code-like
    "The postcode is BH1 1JQ.",
    "Please send it to BH12 3AB.",
    "The office code is A12.",
    "Room A12 is upstairs.",
    "Go to Gate B7.",

    # 7. mixed realistic sentences
    "The Bournemouth Art Gallery opens at 10 AM and is on Westover Road, Bournemouth.",
    "Monet's work is in the Bournemouth Art Gallery.",
    "The Seafood Restaurant is at 12 Westover Road, Bournemouth, England.",
    "John Smith booked a table at The Fishmonger in Bournemouth.",
    "The British Museum in London features works by Turner.",

    # 8. negative / noisy cases
    "I like seafood and art.",
    "This restaurant is nice.",
    "That museum is famous.",
    "A gallery can be interesting.",
    "Fish is served here.",
]

for i, text in enumerate(test_cases, 1):
    print(f"\n=== CASE {i} ===")
    print(text)
    print("-" * 80)

    entities = model.predict_entities(text, labels, threshold=0.65)

    if not entities:
        print("No entities found.")
        continue

    for entity in entities:
        print(f'{entity["text"]} => {entity["label"]}  (score={entity["score"]:.3f})')