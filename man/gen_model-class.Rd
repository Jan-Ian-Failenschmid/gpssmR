\name{gen_model-class}
\docType{class}
\alias{gen_model-class}
\alias{initialize,gen_model-method}
\alias{show,gen_model-method}

\title{Class \code{"gen_model"}}
\description{
%%  ~~ A concise (1-5 lines) description of what the class is. ~~
}
\section{Objects from the Class}{
Objects can be created by calls of the form \code{new("gen_model", model_name, model_type, time, delta, stepsize, pars, start, state_eq, dynamic_error, meas_eq, boundary)}.
%%  ~~ describe objects here ~~ 
}
\section{Slots}{
  \describe{
    \item{\code{model_name}:}{Object of class \code{"character"} ~~ }
    \item{\code{model_type}:}{Object of class \code{"character"} ~~ }
    \item{\code{time}:}{Object of class \code{"numeric"} ~~ }
    \item{\code{delta}:}{Object of class \code{"numeric"} ~~ }
    \item{\code{stepsize}:}{Object of class \code{"numeric"} ~~ }
    \item{\code{pars}:}{Object of class \code{"list"} ~~ }
    \item{\code{start}:}{Object of class \code{"list"} ~~ }
    \item{\code{state_eq}:}{Object of class \code{"list"} ~~ }
    \item{\code{dynamic_error}:}{Object of class \code{"list"} ~~ }
    \item{\code{meas_eq}:}{Object of class \code{"list"} ~~ }
    \item{\code{boundary}:}{Object of class \code{"function"} ~~ }
  }
}
\section{Methods}{
  \describe{
    \item{initialize}{\code{signature(.Object = "gen_model")}: ... }
    \item{show}{\code{signature(object = "gen_model")}: ... }
	 }
}
\references{
%%  ~~put references to the literature/web site here~~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%%  ~~objects to See Also as \code{\link{~~fun~~}}, ~~~
%%  ~~or \code{\linkS4class{CLASSNAME}} for links to other classes ~~~
}
\examples{
showClass("gen_model")
}
\keyword{classes}
