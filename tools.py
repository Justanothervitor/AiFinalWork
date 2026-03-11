import os
from datetime import datetime, timedelta
from typing import Optional, Dict

import gspread
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle
import dotenv
dotenv.load_dotenv()

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets'
]


class GoogleAPIManager:

    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.creds = None
        self.client = None
        self._authenticate()

    def _authenticate(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                self.creds = pickle.load(token)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Please download OAuth2 credentials and save as '{self.credentials_file}'"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                self.creds = flow.run_local_server(port=0)
                self.client = gspread.authorize(self.creds)

            # Save credentials for next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.creds, token)


    def get_calendar_service(self):
        """Get Google Calendar service."""
        return build('calendar', 'v3', credentials=self.creds)

    def get_sheets_service(self):
        """Get Google Sheets service."""
        return build('sheets', 'v4', credentials=self.creds)


class GoogleCalendarTool:
    """Tool for managing Google Calendar bookings."""

    def __init__(self, api_manager: GoogleAPIManager):
        """Initialize Calendar tool with API manager."""
        self.service = api_manager.get_calendar_service()

    def create_booking(self, summary: str, start_time: str, duration_minutes: int = 60,
                       description: str = "", attendee_email: str = None) -> dict:
        try:
            # Parse start time
            start_dt = datetime.fromisoformat(start_time)
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            # Create event
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': 'UTC',
                },
            }

            if attendee_email:
                event['attendees'] = [{'email': attendee_email}]

            # Insert event
            event = self.service.events().insert(calendarId='primary', body=event).execute()

            return {
                'success': True,
                'event_id': event.get('id'),
                'link': event.get('htmlLink'),
                'message': f"Booking created: {summary} on {start_time}"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to create booking: {str(e)}"
            }

    def list_upcoming_events(self, max_results: int = 10) -> dict:

        try:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_list.append({
                    'summary': event.get('summary'),
                    'start': start,
                    'id': event.get('id')
                })

            return {
                'success': True,
                'events': event_list,
                'count': len(event_list)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class GoogleSheetsTool:

    def __init__(self, api_manager: GoogleAPIManager, spreadsheet_id: str = None):
        """Initialize Sheets tool with API manager."""
        self.service = api_manager.get_sheets_service()
        self.spreadsheet_id = spreadsheet_id or os.environ.get('GOOGLE_SPREADSHEET_ID')

    def write_user_data(self, user_name: str, email: str, phone: str = "",address:str="",
                        notes: str = "") -> dict:
        """Write user data to Google Sheets."""
        try:
            if not self.spreadsheet_id:
                return {
                    'success': False,
                    'error': 'No spreadsheet ID configured'
                }

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            values = [[timestamp, user_name, email, phone, address,notes]]

            body = {'values': values}

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Dados!A:E',
                valueInputOption='RAW',
                body=body
            ).execute()

            return {
                'success': True,
                'updated_cells': result.get('updates', {}).get('updatedCells'),
                'message': f"✓ User data recorded for {user_name}"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"✗ Failed to write data: {str(e)}"
            }

    def read_user_data(self, max_rows: int = 100) -> dict:
        """Read recent user data from Google Sheets."""
        try:
            if not self.spreadsheet_id:
                return {
                    'success': False,
                    'error': 'No spreadsheet ID configured'
                }

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f'Dados!A1:E{max_rows + 1}'
            ).execute()

            values = result.get('values', [])

            return {
                'success': True,
                'data': values,
                'count': len(values)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def check_user_exists(self, identifier: str) -> dict:

        try:
            if not self.spreadsheet_id:
                return {
                    'success': False,
                    'exists': False,
                    'error': 'No spreadsheet ID configured'
                }

            # Read all data
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range='Dados!A:E'
            ).execute()

            values = result.get('values', [])

            if not values or len(values) <= 1:  # Only headers or empty
                return {
                    'success': True,
                    'exists': False,
                    'message': 'No users registered yet'
                }

            # Skip header row and search
            identifier_lower = identifier.lower()
            for row in values[1:]:  # Skip header
                if len(row) >= 3:  # Has at least timestamp, name, email
                    name = row[1].lower() if len(row) > 1 else ""
                    email = row[2].lower() if len(row) > 2 else ""

                    if identifier_lower in name or identifier_lower in email:
                        return {
                            'success': True,
                            'exists': True,
                            'user_data': {
                                'name': row[1] if len(row) > 1 else "",
                                'email': row[2] if len(row) > 2 else "",
                                'phone': row[3] if len(row) > 3 else "",
                            }
                        }

            return {
                'success': True,
                'exists': False,
                'message': f'User "{identifier}" not found in records'
            }

        except Exception as e:
            return {
                'success': False,
                'exists': False,
                'error': str(e)
            }
