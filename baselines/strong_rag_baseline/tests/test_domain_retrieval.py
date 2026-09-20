"""Target semantics must retrieve evidence rather than shared words alone."""

from baselines.strong_rag_baseline.tests.test_evidence import _packet


def test_credit_retrieval_keeps_distress_and_liquidity_not_event_boilerplate():
    distress = (
        "Recurring losses and cash funding needs raise substantial doubt about "
        "our ability to continue as a going concern."
    )
    funding = (
        "Sufficient liquidity and borrowing capacity fund operations and "
        "cash obligations; all covenants are in compliance."
    )
    text = "\n".join(
        [
            "In the event that Mr. Smith resigns, compensation is payable.",
            "Credit event historical prior consensus guidance forecasts risks.",
            distress,
            funding,
        ]
    )
    packet, corpus = _packet(
        [("Beta_Company_filing", "2024-05-01", text)],
        target={"name": "credit_event_12m", "type": "classification"},
        top_k=2,
    )
    assert {chunk.text.strip() for chunk in packet.chunks} == {distress, funding}
    for chunk in packet.chunks:
        assert (
            corpus.doc_texts[chunk.doc_id][chunk.span_start : chunk.span_end]
            == chunk.text
        )


def test_earnings_reaction_retrieves_operating_outlook_not_marketer_reaction():
    outlook = (
        "Next quarter revenue and operating income guidance expects growth "
        "in sales and margins; stock price expectations remain volatile."
    )
    packet, _ = _packet(
        [
            (
                "Beta_Company_filing",
                "2024-05-01",
                "Marketer reaction to prior advertising changes was negative.\n"
                "Earnings reaction historical prior reported guidance risks.\n"
                + outlook,
            )
        ],
        target={"name": "earnings_reaction", "type": "classification"},
        top_k=1,
    )
    assert [chunk.text.strip() for chunk in packet.chunks] == [outlook]


def test_customer_credit_growth_does_not_switch_to_solvency():
    packet, _ = _packet(
        [
            (
                "Beta_Company_filing",
                "2024-05-01",
                "Customer credit growth was 12 percent; further growth is forecast.\n"
                "Substantial doubt exists about liquidity and default risk.",
            )
        ],
        target={"name": "customer_credit_growth_pct", "type": "regression"},
        top_k=1,
    )
    assert "12 percent" in packet.chunks[0].text
    assert "Substantial doubt" not in packet.chunks[0].text


def test_non_earnings_returns_do_not_switch_to_company_operating_results():
    packet, _ = _packet(
        [
            (
                "Beta_Company_filing",
                "2024-05-01",
                "Bond return forecasts show 4 percent expected return.\n"
                "Revenue, net income and earnings guidance show sales growth.",
            )
        ],
        target={"name": "bond_return", "type": "regression"},
        top_k=1,
    )
    assert "4 percent" in packet.chunks[0].text
    assert "sales growth" not in packet.chunks[0].text


def test_credit_expansion_does_not_relax_entity_or_cutoff_binding():
    packet, _ = _packet(
        [
            (
                "old",
                "2024-05-01",
                "Beta Company has sufficient liquidity and borrowing capacity.\n"
                "Alpha Company losses raise substantial doubt as a going concern.",
            ),
            (
                "future",
                "2024-06-02",
                "Beta Company default and bankruptcy are now certain.",
            ),
        ],
        target={"name": "credit_event_12m", "type": "classification"},
    )
    assert [chunk.text.strip() for chunk in packet.chunks] == [
        "Beta Company has sufficient liquidity and borrowing capacity."
    ]
