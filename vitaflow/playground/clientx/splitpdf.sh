for f in *.pdf; do pdfseparate ${f} "$1/${f%.*}-%d.pdf"; done

