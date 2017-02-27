class AdapterException(Exception):
    """Generic adapter exception for reporting reading, parsing, and
    converting issues. All adapter exceptions have following instance
    variables in common:

    * `record` - record reference where error occured
    * `field` - field reference where error occured
    """
    def __init__(self, message=None):
    	super(AdapterException, self).__init__(message)

    	self.record = None
    	self.field = None

class SourceError(AdapterException):
	"""Raised on error while reading or talking to a data source. It might be
	seek or read error for file sources or broken connection for database
	sources."""
	pass
		
class SourceNotFoundError(SourceError):
	"""Raised when data source (file, table, ...) was not found."""
	def __init__(self, message=None, source=None):
		super(SourceNotFoundError, self).__init__(message)
		self.source = source

class ConfigurationError(AdapterException):
	"""Raised when objects are mis-configured."""
	pass

class NoSuchFieldError(AdapterException):
	"""Raised when non-existent field is referenced, either by name or position index."""
	pass

class DataIndexError(AdapterException):
	"""Raised for example when a record is not found in record index in indexed
	data source."""
	pass

class DataTypeError(AdapterException):
	"""Raised on data type mis-match or when type conversion fails."""
	pass

class ParserError(AdapterException):
	"""Raised when there is problem with parsing source data, for example in
	broken text file with CSV. The `token` instance variable contains problematic
	token that was not parsed correctly."""
	def __init__(self, message=None, token=None):
		super(ParserError, self).__init__(message)
		self.token = token

class ArgumentError(AdapterException):
	"""Invalid arguments used in calling textadapter functions/methods"""
	pass

class InternalInconsistencyError(AdapterException):
	"""Raised when the library goes into a state that is not expected to
	happen."""
	pass

class AdapterIndexError(AdapterException):
	""" Raised when record number or slice is invalid """
	pass

