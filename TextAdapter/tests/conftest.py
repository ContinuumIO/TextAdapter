import pytest

def pytest_addoption(parser):
    parser.addoption('--pg_host', action='store')
    parser.addoption('--pg_dbname', action='store')
    parser.addoption('--pg_user', action='store')
    parser.addoption('--acc_host', action='store')
    parser.addoption('--acc_user', action='store')
    parser.addoption('--acc_password', action='store')
