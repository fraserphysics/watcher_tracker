(require 'latex)
(condition-case err
    (require 'reftex)
  (error err))

(setq enable-recursive-minibuffers t)
(defun observation-scalar (s time)
  ;; the function "interactive" fetches strings "variable" and "time"
  ;; from the keyboard
  (interactive "svariable:\nsindex:")
  (insert  "\\ti{" s "}{" time "}"))

(define-key LaTeX-mode-map [f1] #'observation-scalar)


(defun observation-vector (s a b e)
  (interactive "svariable:\nsarg:\nsbegin:\nsend:")
  (insert  "\\ts{" s "}{" a "}{" b "}{" e "}"))

(define-key LaTeX-mode-map [f2] #'observation-vector)

