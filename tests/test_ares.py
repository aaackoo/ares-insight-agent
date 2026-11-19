import json
from datetime import date

import httpx
import pytest
import respx

from app.ares import AresClient, AresError
from tests.conftest import ARES_BASE


async def test_get_company_maps_fields(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    company = await ares_client.get_company("27082440")

    assert company is not None
    assert company.name == "Alza.cz a.s."
    assert company.legal_form_code == "121"
    assert company.legal_form == "Akciová společnost"
    assert company.is_active is True
    assert company.founded == date(2003, 8, 26)
    assert company.address == "Jankovcova 1522/53, Holešovice, 17000 Praha 7"
    assert company.source_url == f"{ARES_BASE}/ekonomicke-subjekty/27082440"


async def test_get_company_not_found_returns_none(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    mock_ares.get("/ekonomicke-subjekty/00000001").respond(
        404, json={"kod": "NENALEZENO", "popis": "Záznam nenalezen"}
    )

    assert await ares_client.get_company("1") is None  # also checks zero-padding to 8 digits


async def test_invalid_ico_raises_without_http_call(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    with pytest.raises(AresError, match="Invalid IČO"):
        await ares_client.get_company("ABC123")
    assert mock_ares.calls.call_count == 0


async def test_search_companies(ares_client: AresClient, mock_ares: respx.MockRouter) -> None:
    results = await ares_client.search_companies("Alza", limit=3)

    sent = json.loads(mock_ares.routes["search"].calls.last.request.content)
    assert sent == {"obchodniJmeno": "Alza", "pocet": 3, "start": 0}
    assert [(c.ico, c.name) for c in results] == [
        ("27082440", "Alza.cz a.s."),
        ("46960643", "MS - alza, s.r.o."),
    ]


async def test_statutory_bodies_returns_only_current_members(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    result = await ares_client.get_statutory_bodies("27082440")

    assert result is not None
    assert result.company_name == "Alza.cz a.s."
    assert len(result.bodies) == 1
    body = result.bodies[0]
    assert body.body_name == "Statutární orgán - představenstvo"
    # The fixture has many historical members; only the 4 without a deletion date are current.
    assert len(body.members) == 4
    chair = next(m for m in body.members if m.role == "Předseda představenstva")
    assert chair.name == "ALEŠ ZAVORAL"
    # Role start date, not the 2026-07-23 date when the entry was last re-recorded.
    assert chair.role_since == date(2022, 11, 16)
    assert "Ing. PETR BENA" in [m.name for m in body.members]


async def test_retries_once_on_server_error(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    ok = mock_ares.routes["company"].return_value  # the recorded 200 response
    assert ok is not None
    route = mock_ares.get("/ekonomicke-subjekty/27082440").mock(
        side_effect=[httpx.Response(503), ok]
    )

    company = await ares_client.get_company("27082440")

    assert company is not None
    assert route.call_count == 2


async def test_gives_up_after_second_timeout(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    route = mock_ares.get("/ekonomicke-subjekty/27082440").mock(
        side_effect=httpx.ReadTimeout("slow")
    )

    with pytest.raises(AresError, match="ReadTimeout"):
        await ares_client.get_company("27082440")
    assert route.call_count == 2


async def test_client_error_is_not_retried(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    route = mock_ares.get("/ekonomicke-subjekty/12345678").respond(
        400, json={"kod": "CHYBA_VSTUPU"}
    )

    with pytest.raises(AresError, match="400"):
        await ares_client.get_company("12345678")
    assert route.call_count == 1


async def test_statutory_bodies_for_dissolved_company(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    # Trimmed real VR response for IČO 25010301 (dissolved in 2025). Every entry has a
    # deletion date, and the basic /ekonomicke-subjekty endpoint returns 404 for it.
    mock_ares.get("/ekonomicke-subjekty-vr/25010301").respond(
        json={
            "icoId": "25010301",
            "zaznamy": [
                {
                    "primarniZaznam": True,
                    "stavSubjektu": "ZANIKLY",
                    "datumVymazu": "2025-07-22",
                    "obchodniJmeno": [
                        {
                            "datumZapisu": "1996-10-17",
                            "datumVymazu": "2025-07-22",
                            "hodnota": "PIZZA PARTY, s.r.o.",
                        }
                    ],
                    "statutarniOrgany": [
                        {"nazevOrganu": "Statutární orgán", "datumVymazu": "2025-07-22"}
                    ],
                }
            ],
        }
    )

    result = await ares_client.get_statutory_bodies("25010301")

    assert result is not None
    assert result.company_name == "PIZZA PARTY, s.r.o."  # last known name
    assert result.status == "ZANIKLY"
    assert result.deleted_from_register == date(2025, 7, 22)
    assert result.bodies == []


async def test_search_skips_entries_without_ico(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    # Real case (search "Kofola"): a branch ("odštěpný závod") has no IČO of its own.
    mock_ares.post("/ekonomicke-subjekty/vyhledat").respond(
        json={
            "pocetCelkem": 2,
            "ekonomickeSubjekty": [
                {"obchodniJmeno": "Kofola a.s., divize výroba a obchod - odštěpný závod"},
                {"ico": "27767680", "obchodniJmeno": "Kofola a.s.", "pravniForma": "121"},
            ],
        }
    )

    results = await ares_client.search_companies("Kofola")

    assert [c.ico for c in results] == ["27767680"]
