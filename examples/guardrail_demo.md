# Guardrail demo: the citation check catching a hallucination

A scripted model drafted three sentences. Only the first one is in the policy.

**What the model drafted:**

- The extended warranty costs 8% of the device's selling price and brings total cover to 36 months for laptops and desktops. `['S1']`
- It also includes free accidental damage cover for 24 months. `['S1']`
- The extended warranty costs 5% for orders above MYR 50,000. `['S1']`

**What the user sees:**

**Q:** How much does the extended warranty cost and what does it include?

✅ Answered · confidence 0.73 (medium) · source: knowledge base

The extended warranty costs 8% of the device's selling price and brings total cover to 36 months for laptops and desktops. [1]

**Sources**
1. Warranty and Returns Policy › Extended warranty (`warranty_and_returns#extended-warranty`)

**Removed by the citation check**
- ~~It also includes free accidental damage cover for 24 months.~~ (number(s) not in the cited source: 24; only 43% of its key words appear in the cited source)
- ~~The extended warranty costs 5% for orders above MYR 50,000.~~ (number(s) not in the cited source: 5%, 50000; only 43% of its key words appear in the cited source)

_Why this confidence: 2 of 3 drafted sentence(s) failed the citation check and were removed_
