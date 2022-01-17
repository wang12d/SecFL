update:
	@ git add .
	@ git commit -am "update"

clean:
	@ find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

.PHONY : submit
submit:
	@ git push origin