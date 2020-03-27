"%PYTHON%"  setup.py clean --all install --prefix="%PREFIX%"
if errorlevel 1 exit 1