"""The three ARES tools the agent can call.

Tools are built by a function (not module-level globals) so each one closes over an
AresClient that the caller provides: the real one in the app, a respx-mocked one in tests.

Each tool returns a JSON string. The LLM reads it, including every record's `source_url`,
which it must copy into the final answer. Errors are returned as text, not raised, so the
agent can react (e.g. try a search when an IČO is wrong) instead of the whole run failing.
"""

from langchain_core.tools import BaseTool, tool
from pydantic import TypeAdapter

from app.ares import AresClient, AresError, Company


def build_tools(client: AresClient) -> list[BaseTool]:
    @tool
    async def get_company_by_ico(ico: str) -> str:
        """Get a Czech company's basic record from ARES by its IČO (8-digit company ID).

        Returns name, legal form, registered address, founding date, dissolution date,
        whether it is active, and the source URL. Dissolved companies are not returned.
        """
        try:
            company = await client.get_company(ico)
        except AresError as exc:
            return f"Error: {exc}"
        if company is None:
            # ARES returns 404 here for dissolved companies too, so steer the agent to VR
            # instead of letting it conclude "this company never existed".
            return (
                f"No active company found in ARES with IČO {ico}. It may be dissolved: "
                "call get_statutory_bodies, the commercial register also keeps dissolved "
                "companies."
            )
        return company.model_dump_json()

    @tool
    async def search_companies_by_name(name: str) -> str:
        """Search Czech companies in ARES by business name (or part of it).

        Use this when the user gives a company name but no IČO. Returns up to 5 matches,
        each with its IČO, so you can call the other tools afterwards.
        """
        try:
            companies = await client.search_companies(name, limit=5)
        except AresError as exc:
            return f"Error: {exc}"
        if not companies:
            return f"No companies found in ARES matching {name!r}."
        return _companies.dump_json(companies).decode()

    @tool
    async def get_statutory_bodies(ico: str) -> str:
        """Get a company's commercial register record: status and current statutory bodies.

        Statutory bodies are the people (or entities) who legally represent the company,
        e.g. the board of directors (představenstvo) or executives (jednatelé).
        Returns the register status (AKTIVNI = active, ZANIKLY = dissolved), the date it
        was deleted from the register if dissolved, each body with its current members
        and their roles, and the source URL. Also works for dissolved companies.
        """
        try:
            result = await client.get_statutory_bodies(ico)
        except AresError as exc:
            return f"Error: {exc}"
        if result is None:
            return (
                f"No commercial register record for IČO {ico}. The company may not exist "
                "or may not be registered there (e.g. a sole trader)."
            )
        return result.model_dump_json()

    return [get_company_by_ico, search_companies_by_name, get_statutory_bodies]


# Serializes a list of Pydantic models to JSON in one call.
_companies = TypeAdapter(list[Company])
