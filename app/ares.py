"""Async client for the public ARES REST API.

API docs: https://ares.gov.cz/swagger-ui/  (OpenAPI: {base}/v3/api-docs)
Endpoints used:
  GET  /ekonomicke-subjekty/{ico}       basic company record
  POST /ekonomicke-subjekty/vyhledat    search by name
  GET  /ekonomicke-subjekty-vr/{ico}    commercial register (statutory bodies)
"""

import logging
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Subset of the official ARES code list "PravniForma" (verified against
# POST /ciselniky-nazevniky/vyhledat). Unknown codes are returned as the code only.
LEGAL_FORMS = {
    "101": "Fyzická osoba podnikající dle živnostenského zákona",
    "111": "Veřejná obchodní společnost",
    "112": "Společnost s ručením omezeným",
    "113": "Společnost komanditní",
    "121": "Akciová společnost",
    "141": "Obecně prospěšná společnost",
    "205": "Družstvo",
    "301": "Státní podnik",
    "706": "Spolek",
}


class AresError(Exception):
    """ARES returned an error, was unreachable, or the input was invalid."""


class Company(BaseModel):
    ico: str
    name: str
    legal_form_code: str | None
    legal_form: str | None
    address: str | None
    founded: date | None
    dissolved: date | None
    is_active: bool
    source_url: str


class StatutoryMember(BaseModel):
    name: str
    role: str | None
    role_since: date | None  # when the person took this role (not when it was recorded)


class StatutoryBody(BaseModel):
    body_name: str
    members: list[StatutoryMember]


class StatutoryBodies(BaseModel):
    ico: str
    company_name: str
    status: str | None  # VR "stavSubjektu", e.g. AKTIVNI (active) or ZANIKLY (dissolved)
    deleted_from_register: date | None
    bodies: list[StatutoryBody]
    source_url: str


def normalize_ico(ico: str) -> str:
    """IČO is 8 digits; shorter ones are left-padded with zeros (e.g. '177041')."""
    cleaned = ico.replace(" ", "").strip()
    if not cleaned.isdigit() or len(cleaned) > 8:
        raise AresError(f"Invalid IČO {ico!r}: expected up to 8 digits.")
    return cleaned.zfill(8)


class AresClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        # The httpx client is injected so the app owns its lifecycle and timeouts,
        # and tests can pass a client whose transport is mocked by respx.
        self._http = http
        self._base_url = base_url.rstrip("/")

    def company_url(self, ico: str) -> str:
        return f"{self._base_url}/ekonomicke-subjekty/{ico}"

    def commercial_register_url(self, ico: str) -> str:
        return f"{self._base_url}/ekonomicke-subjekty-vr/{ico}"

    async def get_company(self, ico: str) -> Company | None:
        """Return the company, or None if not found.

        Note: this endpoint returns 404 for dissolved companies too; only the commercial
        register (get_statutory_bodies) still has those.
        """
        ico = normalize_ico(ico)
        url = self.company_url(ico)
        data = await self._request("GET", url)
        return None if data is None else self._to_company(data)

    async def search_companies(self, name: str, limit: int = 5) -> list[Company]:
        """Search companies by (part of) their business name."""
        url = f"{self._base_url}/ekonomicke-subjekty/vyhledat"
        body = {"obchodniJmeno": name, "pocet": limit, "start": 0}
        data = await self._request("POST", url, body)
        if data is None:
            return []
        # Skip entries without an IČO (e.g. branches, "odštěpný závod"): the other
        # tools need an IČO, so they would be dead ends for the agent.
        items = data.get("ekonomickeSubjekty", [])
        return [self._to_company(item) for item in items if item.get("ico")]

    async def get_statutory_bodies(self, ico: str) -> StatutoryBodies | None:
        """Return status and current statutory bodies from the commercial register (VR).

        Also covers dissolved companies (status ZANIKLY, with no current bodies).
        None means the company is not in the commercial register
        (e.g. a sole trader), or the IČO does not exist.
        """
        ico = normalize_ico(ico)
        url = self.commercial_register_url(ico)
        data = await self._request("GET", url)
        if data is None:
            return None
        # A company may have several VR records; the primary one is the current one.
        records = data.get("zaznamy", [])
        record = next((r for r in records if r.get("primarniZaznam")), None)
        if record is None:
            return None

        bodies = []
        for body in _current(record.get("statutarniOrgany", [])):
            members = [self._to_member(m) for m in _current(body.get("clenoveOrganu", []))]
            if members:  # skip bodies whose members are all historical
                bodies.append(StatutoryBody(body_name=body.get("nazevOrganu", ""), members=members))
        # A dissolved company has no current name; fall back to its last recorded one.
        names = _current(record.get("obchodniJmeno", [])) or record.get("obchodniJmeno", [])[-1:]
        return StatutoryBodies(
            ico=ico,
            company_name=names[0]["hodnota"] if names else "",
            status=record.get("stavSubjektu"),
            deleted_from_register=record.get("datumVymazu"),
            bodies=bodies,
            source_url=url,
        )

    async def _request(
        self, method: str, url: str, body: dict[str, Any] | None = None
    ) -> Any | None:
        """Call ARES and return parsed JSON, or None on 404."""
        try:
            response = await self._send(method, url, body)
        except httpx.HTTPError as exc:  # timeouts, connection errors
            raise AresError(f"ARES request failed: {type(exc).__name__}") from exc

        if response.status_code == 404:
            return None
        if response.is_error:
            # ARES error bodies look like {"kod": ..., "popis": "human readable message"}
            raise AresError(f"ARES returned {response.status_code}: {response.text[:300]}")
        return response.json()

    async def _send(self, method: str, url: str, body: dict[str, Any] | None) -> httpx.Response:
        """Send the request, retrying exactly once on timeout or 5xx."""
        try:
            response = await self._http.request(method, url, json=body)
            if response.status_code < 500:
                return response
            logger.warning("ARES %s %s -> %s, retrying once", method, url, response.status_code)
        except httpx.TimeoutException:
            logger.warning("ARES %s %s timed out, retrying once", method, url)
        # The retry: if it fails as well, its response / exception is what the caller sees.
        return await self._http.request(method, url, json=body)

    def _to_company(self, data: dict[str, Any]) -> Company:
        ico = data["ico"]
        code = data.get("pravniForma")
        dissolved = data.get("datumZaniku")
        return Company(
            ico=ico,
            name=data.get("obchodniJmeno", ""),
            legal_form_code=code,
            legal_form=LEGAL_FORMS.get(code) if code else None,
            address=(data.get("sidlo") or {}).get("textovaAdresa"),
            founded=data.get("datumVzniku"),
            dissolved=dissolved,
            is_active=dissolved is None,
            source_url=self.company_url(ico),
        )

    @staticmethod
    def _to_member(data: dict[str, Any]) -> StatutoryMember:
        person = data.get("fyzickaOsoba")
        if person:
            parts = [person.get("titulPredJmenem"), person.get("jmeno"), person.get("prijmeni")]
            name = " ".join(p for p in parts if p)
        else:  # a legal entity can also be a member of a statutory body
            name = (data.get("pravnickaOsoba") or {}).get("obchodniJmeno", "unknown")
        # Not "datumZapisu": that is when the entry was (re)recorded in the register,
        # which can be years after the person actually took the role.
        membership = data.get("clenstvi") or {}
        function = membership.get("funkce") or {}
        role_since = function.get("vznikFunkce") or (membership.get("clenstvi") or {}).get(
            "vznikClenstvi"
        )
        return StatutoryMember(name=name, role=function.get("nazev"), role_since=role_since)


def _current(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """VR keeps full history; entries without 'datumVymazu' (deletion date) are current."""
    return [e for e in entries if "datumVymazu" not in e]
