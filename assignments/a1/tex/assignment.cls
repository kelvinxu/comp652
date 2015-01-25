\LoadClass{article}

\renewcommand{\normalsize}{\fontsize{10pt}{13pt}\selectfont}

% Default paper size letter
\RequirePackage[letterpaper, margin=2.54cm]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{etoolbox}

% commands to set title and header block info
\def\title#1{\gdef\@title{#1}\gdef\stored@title{#1}}
\def\author#1{\gdef\@author{#1}\gdef\stored@author{#1}}
\def\date#1{\gdef\@date{#1}\gdef\stored@date{#1}}
%\newcommand\studentnumber[1]{\renewcommand\@studentnumber{#1}}
%\newcommand\@studentnumber{\@latex@error{No \noexpand\studentnumber given}\@ehc}
\newcommand\course[1]{\renewcommand\@course{#1}}
\newcommand\@course{\@latex@error{No \noexpand\course given}\@ehc}

\pagestyle{fancy}
\fancyhf{} % clear the commands
\lhead{\it \stored@author}
\chead{\it \@course : \stored@title}
\rhead{\it \stored@date}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.01pt}

\RequirePackage{amsmath}
\RequirePackage{amssymb}

% Title bars
\def\toptitlebar{\hrule height4pt\vskip .25in\vskip-\parskip}
\def\bottomtitlebar{\vskip .29in\vskip-\parskip\hrule height1pt\vskip.09in} %

\def\@maketitle{\vbox{\hsize\textwidth
    \linewidth\hsize \vskip 0.1in \toptitlebar \vskip 0.1in\centering
    {\LARGE\bf\@course : \@title\par}  \bottomtitlebar %   minus
    \def\And{\end{tabular}\hfil\linebreak[0]\hfil
        \begin{tabular}[t]{c}\bf\rule{\z@}{24pt}\ignorespaces}% 
    \def\AND{\end{tabular}\hfil\linebreak[4]\hfil
        \begin{tabular}[t]{c}\bf\rule{\z@}{24pt}\ignorespaces}% 
    \begin{tabular}[t]{c}
        \\\bf\rule{\z@}{24pt}
        \large{\@author} \\ % $\mid$ \large{\@studentnumber} \\ 
        % ADD THINGS HERE TO ADD TO THE TITLE BOX
        \\
        \@date\end{tabular}% 
\vskip 0.1in minus 0.2in}}

% Allow TeX to put a widowed line at the bottom of previous page
\flushbottom \sloppy

% Sections with less space
\def\section{\@startsection {section}{1}{\z@}{-2.0ex plus
    -0.5ex minus -.2ex}{1.5ex plus 0.3ex
    minus0.2ex}{\large\bf\raggedright}}
\def\subsection{\@startsection{subsection}{2}{\z@}{-1.8ex plus    
    -0.5ex minus -.2ex}{0.8ex plus .2ex}{\normalsize\bf\raggedright}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-1.5ex
    plus -0.5ex minus -.2ex}{0.5ex plus
    .2ex}{\normalsize\bf\raggedright}}
\def\subsubsubsection{\vskip
    5pt{\noindent\normalsize\rm\raggedright}}
\def\paragraph{\@startsection{paragraph}{4}{\z@}{1.5ex plus   
    0.5ex minus .2ex}{-1em}{\normalsize\bf}}
\def\subparagraph{\@startsection{subparagraph}{5}{\z@}{1.5ex plus 
    0.5ex minus .2ex}{-1em}{\normalsize\bf}}

% problem-solution formatting
% Set up counters for problems and subsections
 
\newcounter{ProblemNum}
\newcounter{SubProblemNum}[ProblemNum]
 
\renewcommand{\theProblemNum}{\arabic{ProblemNum}}
\renewcommand{\theSubProblemNum}{\alph{SubProblemNum}}
 
\newcommand*{\anyproblem}[1]{\subsection*{#1}}
\newcommand*{\problem}[1]{\stepcounter{ProblemNum} %
\anyproblem{Problem \theProblemNum\vskip.04in\hrule\vskip.09in #1}}
\renewcommand*{\part}[1]{\stepcounter{SubProblemNum} %
\noindent\textbf{\hspace*{15pt}Part (\theSubProblemNum)} \; \hangindent=15pt #1 \vspace{5pt}}

\newenvironment{solution}{\hspace*{15pt}\textbf{Solution} \; \hangindent=30pt \vspace{10pt}}{}
 
%\renewcommand{\theenumi}{(\alph{enumi})}
%\renewcommand{\labelenumi}{\theenumi}
%\renewcommand{\theenumii}{\roman{enumii}}
